"""Admin API routes"""
import asyncio
import io
import json
import re
import secrets
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from curl_cffi.requests import AsyncSession

from ..core.auth import AuthManager
from ..core.config import config
from ..core.database import Database
from ..core.models import normalize_captcha_priority_order
from ..services.concurrency_manager import ConcurrencyManager
from ..services.load_balancer import LoadBalancer
from ..services.perf_monitor import perf_monitor
from ..services.proxy_manager import ProxyManager
from ..services.token_manager import TokenManager

router = APIRouter()

# Dependency injection
token_manager: TokenManager = None
proxy_manager: ProxyManager = None
db: Database = None
concurrency_manager: Optional[ConcurrencyManager] = None
load_balancer: Optional[LoadBalancer] = None
cluster_manager = None

# Store active admin session tokens (in production, use Redis or database)
active_admin_tokens = set()
SUPPORTED_API_CAPTCHA_METHODS = {"yescaptcha", "capmonster", "ezcaptcha", "capsolver"}
admin_bearer = HTTPBearer(
    scheme_name="AdminSessionBearer",
    description="管理后台会话鉴权。先调用 /api/admin/login 获取后台 session token，再以 Bearer 方式传入。",
    auto_error=False,
)


def _mask_token(token: Optional[str]) -> str:
    if not token:
        return ""
    if len(token) <= 24:
        return token
    return f"{token[:18]}...{token[-8:]}"


HARD_BAN_REASON_CODES = {
    "429_rate_limit",
    "permission_denied",
    "google_account_disabled",
}


def _parse_optional_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _format_disable_reason_label(reason_code: str) -> str:
    code = str(reason_code or "").strip()
    reason_map = {
        "cookie_invalid_need_relogin": "Cookie失效待重登",
        "429_rate_limit": "429频率封禁",
        "permission_denied": "403权限封禁",
        "google_account_disabled": "Google账号封禁",
        "manual_disabled": "手动禁用",
        "import_disabled": "导入时设为禁用",
        "token_expired": "Token已过期",
        "refresh_failed": "刷新失败后自动禁用",
        "consecutive_error_limit": "连续错误达到阈值自动禁用",
        "unknown_inactive": "已禁用(原因未记录)",
    }
    return reason_map.get(code, code or "未禁用")


def _build_disable_reason_meta(row: Dict[str, Any], error_ban_threshold: int) -> Dict[str, Any]:
    is_active = bool(row.get("is_active"))
    reason_code = str(row.get("ban_reason") or "").strip()
    last_refresh_detail = str(row.get("last_refresh_detail") or "").strip()
    at_expires = _parse_optional_datetime(row.get("at_expires"))
    banned_at = _parse_optional_datetime(row.get("banned_at"))
    last_error_at = _parse_optional_datetime(row.get("last_error_at"))
    consecutive_error_count = int(row.get("consecutive_error_count") or 0)

    if not is_active and not reason_code:
        now = datetime.now(timezone.utc)
        if at_expires:
            exp_aware = at_expires if at_expires.tzinfo else at_expires.replace(tzinfo=timezone.utc)
            if exp_aware <= now:
                reason_code = "token_expired"
        detail_lower = last_refresh_detail.lower()
        if not reason_code and ("自动禁用" in detail_lower or "auto-disable" in detail_lower):
            reason_code = "refresh_failed"
        if not reason_code and error_ban_threshold > 0 and consecutive_error_count >= error_ban_threshold:
            reason_code = "consecutive_error_limit"
        if not reason_code:
            reason_code = "unknown_inactive"

    label = _format_disable_reason_label(reason_code) if reason_code else ""
    detail = ""
    if reason_code == "manual_disabled":
        detail = "管理员手动禁用了该 Token"
    elif reason_code == "import_disabled":
        detail = "导入或更新 Token 时被标记为禁用"
    elif reason_code == "token_expired":
        detail = "Token 当前 AT 已过期，因此处于禁用状态"
    elif reason_code == "refresh_failed":
        detail = last_refresh_detail or "ST直刷/reAuth/浏览器刷新均失败后自动禁用"
    elif reason_code == "consecutive_error_limit":
        if error_ban_threshold > 0:
            detail = (
                f"连续错误次数 {consecutive_error_count} 已达到自动禁用阈值 "
                f"{error_ban_threshold}"
            )
        else:
            detail = "连续错误次数达到自动禁用阈值"
    elif reason_code == "cookie_invalid_need_relogin":
        detail = last_refresh_detail or "Cookie 已失效，需要重新自动登录恢复"
    elif reason_code == "429_rate_limit":
        detail = "请求触发 429 频率限制，系统已自动停用"
    elif reason_code == "permission_denied":
        detail = "请求命中 403 PERMISSION_DENIED，系统判定账号权限被封禁"
    elif reason_code == "google_account_disabled":
        detail = last_refresh_detail or "账号池/RPA 检测到 Google 账号已被停用或封禁"
    elif reason_code == "unknown_inactive":
        detail = "该 Token 当前处于禁用状态，但历史数据未记录明确原因"

    effective_time = banned_at or last_error_at
    return {
        "disable_reason_code": reason_code or None,
        "disable_reason_label": label or None,
        "disable_reason_detail": detail or None,
        "disable_hard_banned": bool(reason_code and reason_code in HARD_BAN_REASON_CODES),
        "disable_effective_at": effective_time.isoformat() if effective_time else None,
    }


def _normalize_project_action_weight(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = 1
    return max(1, parsed)


def _build_code_service_project_items(token_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in token_rows:
        project_id = str(row.get("current_project_id") or "").strip()
        if not project_id:
            continue

        token_id_raw = row.get("id")
        try:
            token_id = int(token_id_raw) if token_id_raw is not None else None
        except Exception:
            token_id = None

        is_token_enabled = bool(row.get("is_active"))

        if bool(row.get("video_enabled")):
            items.append(
                {
                    "project_id": project_id,
                    "action": "VIDEO_GENERATION",
                    "token_id": token_id,
                    "weight": _normalize_project_action_weight(row.get("video_concurrency")),
                    "enabled": is_token_enabled,
                }
            )

        if bool(row.get("image_enabled")):
            items.append(
                {
                    "project_id": project_id,
                    "action": "IMAGE_GENERATION",
                    "token_id": token_id,
                    "weight": _normalize_project_action_weight(row.get("image_concurrency")),
                    "enabled": is_token_enabled,
                }
            )

    return items


def _guess_client_hints_from_user_agent(user_agent: str) -> Dict[str, str]:
    """根据 UA 补全常见的 sec-ch-* 头。"""
    ua = (user_agent or "").strip()
    if not ua:
        return {}

    headers: Dict[str, str] = {}
    major_match = re.search(r"(?:Chrome|Chromium|Edg|EdgA|EdgiOS)/(\d+)", ua)
    is_mobile = any(token in ua for token in ("Android", "iPhone", "iPad", "Mobile"))
    headers["sec-ch-ua-mobile"] = "?1" if is_mobile else "?0"

    if "Windows" in ua:
        headers["sec-ch-ua-platform"] = '"Windows"'
    elif "Macintosh" in ua or "Mac OS X" in ua:
        headers["sec-ch-ua-platform"] = '"macOS"'
    elif "Android" in ua:
        headers["sec-ch-ua-platform"] = '"Android"'
    elif "iPhone" in ua or "iPad" in ua:
        headers["sec-ch-ua-platform"] = '"iOS"'
    elif "Linux" in ua:
        headers["sec-ch-ua-platform"] = '"Linux"'

    if major_match:
        major = major_match.group(1)
        if "Edg/" in ua:
            headers["sec-ch-ua"] = (
                f'"Not:A-Brand";v="99", "Microsoft Edge";v="{major}", "Chromium";v="{major}"'
            )
        else:
            headers["sec-ch-ua"] = (
                f'"Not:A-Brand";v="99", "Google Chrome";v="{major}", "Chromium";v="{major}"'
            )

    return headers


def _guess_impersonate_from_user_agent(user_agent: str) -> str:
    """从 UA 选择可用的 curl_cffi 浏览器指纹版本。"""
    ua = (user_agent or "").strip()
    major_match = re.search(r"(?:Chrome|Chromium|Edg|EdgA|EdgiOS)/(\d+)", ua)
    if not major_match:
        return "chrome120"

    try:
        major = int(major_match.group(1))
    except Exception:
        return "chrome120"

    if major >= 124:
        return "chrome124"
    if major >= 120:
        return "chrome120"
    return "chrome120"


def _build_proxy_map(proxy_url: str) -> Optional[Dict[str, str]]:
    normalized = (proxy_url or "").strip()
    if not normalized:
        return None
    return {"http": normalized, "https": normalized}


def _normalize_http_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("远程打码服务地址未配置")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("远程打码服务地址格式错误，必须是 http(s)://host[:port]")

    return normalized


def _get_remote_browser_client_config() -> tuple[str, str, int]:
    base_url = _normalize_http_base_url(config.remote_browser_base_url)
    api_key = (config.remote_browser_api_key or "").strip()
    if not api_key:
        raise RuntimeError("远程打码服务 API Key 未配置")
    timeout = max(5, int(config.remote_browser_timeout or 60))
    return base_url, api_key, timeout


def _sync_json_http_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]],
    timeout: int,
) -> tuple[int, Optional[Any], str]:
    req_headers = dict(headers or {})
    req_headers.setdefault("Accept", "application/json")

    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=req_headers,
        method=(method or "GET").upper(),
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.getcode() or 0)
            raw_body = response.read()
    except urllib.error.HTTPError as e:
        status_code = int(getattr(e, "code", 500))
        raw_body = e.read() if hasattr(e, "read") else b""
    except Exception as e:
        raise RuntimeError(f"远程打码服务请求失败: {e}") from e

    text = raw_body.decode("utf-8", errors="replace") if raw_body else ""
    parsed: Optional[Any] = None
    if text:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    return status_code, parsed, text


async def _resolve_score_test_verify_proxy(
    captcha_method: str,
    browser_proxy_enabled: bool,
    browser_proxy_url: str
) -> tuple[Optional[Dict[str, str]], bool, str, str]:
    """
    选择 score-test 的 verify 请求代理，优先与浏览器打码代理保持一致。
    返回: (proxies, used, source, proxy_url)
    """
    # 浏览器打码模式优先使用 browser_proxy，确保与取 token 出口一致
    if captcha_method in {"browser", "personal"} and browser_proxy_enabled and browser_proxy_url:
        proxy_map = _build_proxy_map(browser_proxy_url)
        if proxy_map:
            return proxy_map, True, "captcha_browser_proxy", browser_proxy_url

    # 退回请求代理配置
    try:
        if proxy_manager:
            proxy_cfg = await proxy_manager.get_proxy_config()
            if proxy_cfg and proxy_cfg.enabled and proxy_cfg.proxy_url:
                proxy_map = _build_proxy_map(proxy_cfg.proxy_url)
                if proxy_map:
                    return proxy_map, True, "request_proxy", proxy_cfg.proxy_url
    except Exception:
        pass

    return None, False, "none", ""


async def _solve_recaptcha_with_api_service(
    method: str,
    website_url: str,
    website_key: str,
    action: str,
    enterprise: bool = False
) -> Optional[str]:
    """使用当前配置的第三方打码服务获取 token。"""
    if method == "yescaptcha":
        client_key = config.yescaptcha_api_key
        base_url = config.yescaptcha_base_url
        task_type = "RecaptchaV3TaskProxylessM1"
    elif method == "capmonster":
        client_key = config.capmonster_api_key
        base_url = config.capmonster_base_url
        task_type = "RecaptchaV3TaskProxyless"
    elif method == "ezcaptcha":
        client_key = config.ezcaptcha_api_key
        base_url = config.ezcaptcha_base_url
        task_type = "ReCaptchaV3TaskProxylessS9"
    elif method == "capsolver":
        client_key = config.capsolver_api_key
        base_url = config.capsolver_base_url
        task_type = "ReCaptchaV3EnterpriseTaskProxyLess" if enterprise else "ReCaptchaV3TaskProxyLess"
    else:
        raise RuntimeError(f"不支持的打码方式: {method}")

    if not client_key:
        raise RuntimeError(f"{method} API Key 未配置")

    task: Dict[str, Any] = {
        "websiteURL": website_url,
        "websiteKey": website_key,
        "type": task_type,
        "pageAction": action,
    }

    if enterprise and method == "capsolver":
        task["isEnterprise"] = True

    create_url = f"{base_url.rstrip('/')}/createTask"
    get_url = f"{base_url.rstrip('/')}/getTaskResult"

    async with AsyncSession() as session:
        create_resp = await session.post(
            create_url,
            json={"clientKey": client_key, "task": task},
            impersonate="chrome120",
            timeout=30
        )
        create_json = create_resp.json()
        task_id = create_json.get("taskId")

        if not task_id:
            error_desc = create_json.get("errorDescription") or create_json.get("errorMessage") or str(create_json)
            raise RuntimeError(f"{method} createTask 失败: {error_desc}")

        for _ in range(40):
            poll_resp = await session.post(
                get_url,
                json={"clientKey": client_key, "taskId": task_id},
                impersonate="chrome120",
                timeout=30
            )
            poll_json = poll_resp.json()
            if poll_json.get("status") == "ready":
                solution = poll_json.get("solution", {}) or {}
                token = solution.get("gRecaptchaResponse") or solution.get("token")
                if token:
                    return token
                raise RuntimeError(f"{method} 返回结果缺少 token: {poll_json}")

            if poll_json.get("errorId") not in (None, 0):
                error_desc = poll_json.get("errorDescription") or poll_json.get("errorMessage") or str(poll_json)
                raise RuntimeError(f"{method} getTaskResult 失败: {error_desc}")

            await asyncio.sleep(3)

    raise RuntimeError(f"{method} 获取 token 超时")


async def _score_test_with_remote_browser_service(
    website_url: str,
    website_key: str,
    verify_url: str,
    action: str,
    enterprise: bool = False,
) -> Dict[str, Any]:
    """调用远程有头打码服务执行页面内打码+分数校验。"""
    base_url, api_key, timeout = _get_remote_browser_client_config()
    endpoint = f"{base_url}/api/v1/custom-score"
    request_payload = {
        "website_url": website_url,
        "website_key": website_key,
        "verify_url": verify_url,
        "action": action,
        "enterprise": enterprise,
    }

    status_code, response_payload, response_text = await asyncio.to_thread(
        _sync_json_http_request,
        "POST",
        endpoint,
        {"Authorization": f"Bearer {api_key}"},
        request_payload,
        timeout,
    )

    if status_code >= 400:
        detail = ""
        if isinstance(response_payload, dict):
            detail = response_payload.get("detail") or response_payload.get("message") or str(response_payload)
        if not detail:
            detail = (response_text or "").strip()
        raise RuntimeError(f"远程打码服务请求失败 (HTTP {status_code}): {detail or '未知错误'}")

    if not isinstance(response_payload, dict):
        raise RuntimeError("远程打码服务返回格式错误")
    return response_payload


def set_dependencies(
    tm: TokenManager,
    pm: ProxyManager,
    database: Database,
    cm: Optional[ConcurrencyManager] = None,
    lb: Optional[LoadBalancer] = None,
    cluster_mgr = None,
):
    """Set service instances"""
    global token_manager, proxy_manager, db, concurrency_manager, load_balancer, cluster_manager
    token_manager = tm
    proxy_manager = pm
    db = database
    concurrency_manager = cm
    load_balancer = lb
    cluster_manager = cluster_mgr


# ========== Request Models ==========

class LoginRequest(BaseModel):
    username: str
    password: str


class AddTokenRequest(BaseModel):
    st: str
    cookie: Optional[str] = None
    cookie_file: Optional[str] = None
    project_id: Optional[str] = None  # 用户可选输入project_id
    project_name: Optional[str] = None
    remark: Optional[str] = None
    captcha_proxy_url: Optional[str] = None
    image_enabled: bool = True
    video_enabled: bool = True
    image_concurrency: int = -1
    video_concurrency: int = -1


class UpdateTokenRequest(BaseModel):
    st: str  # Session Token (必填，用于刷新AT)
    cookie: Optional[str] = None
    cookie_file: Optional[str] = None
    project_id: Optional[str] = None  # 用户可选输入project_id
    project_name: Optional[str] = None
    remark: Optional[str] = None
    captcha_proxy_url: Optional[str] = None
    image_enabled: Optional[bool] = None
    video_enabled: Optional[bool] = None
    image_concurrency: Optional[int] = None
    video_concurrency: Optional[int] = None


class ProxyConfigRequest(BaseModel):
    proxy_enabled: bool
    proxy_url: Optional[str] = None
    media_proxy_enabled: Optional[bool] = None
    media_proxy_url: Optional[str] = None


class ProxyTestRequest(BaseModel):
    proxy_url: str
    test_url: Optional[str] = "https://labs.google/"
    timeout_seconds: Optional[int] = 15


class CaptchaScoreTestRequest(BaseModel):
    website_url: Optional[str] = "https://antcpt.com/score_detector/"
    website_key: Optional[str] = "6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf"
    action: Optional[str] = "homepage"
    verify_url: Optional[str] = "https://antcpt.com/score_detector/verify.php"
    enterprise: Optional[bool] = False


class ServerConfigRequest(BaseModel):
    mode: str  # local | server
    host: Optional[str] = None
    port: Optional[int] = None
    default_public_ip: Optional[str] = None
    rpa_test_bitbrowser_id_local: Optional[str] = None
    rpa_test_bitbrowser_id_server: Optional[str] = None
    rpa_test_bitbrowser_ids_local: Optional[List[str]] = None
    rpa_test_bitbrowser_ids_server: Optional[List[str]] = None
    block_gemini_25_flash_image: Optional[bool] = None


class ClusterConfigRequest(BaseModel):
    role: str  # standalone | master | worker
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    master_base_url: Optional[str] = None
    node_public_base_url: Optional[str] = None
    cluster_key: Optional[str] = None
    node_weight: Optional[int] = None
    node_max_concurrency: Optional[int] = None
    heartbeat_interval_seconds: Optional[int] = None
    heartbeat_timeout_seconds: Optional[int] = None
    dispatch_timeout_seconds: Optional[int] = None
    prefer_local: Optional[bool] = None


class ClusterHeartbeatRequest(BaseModel):
    node_id: str
    node_name: Optional[str] = None
    base_url: Optional[str] = None
    role: Optional[str] = "worker"
    enabled: bool = True
    active_requests: int = 0
    node_max_concurrency: int = 0
    available_slots: Optional[int] = None
    weight: int = 100
    reported_roundtrip_ms: Optional[int] = None
    reported_at: Optional[str] = None
    load_summary: Dict[str, Any] = Field(default_factory=dict)


class ClusterDeleteNodeRequest(BaseModel):
    node_id: str


class GenerationConfigRequest(BaseModel):
    image_timeout: int
    video_timeout: int


class LogCleanupRequest(BaseModel):
    older_than_days: Optional[int] = 7
    keep_latest: Optional[int] = 5000
    trim_payloads: bool = True
    vacuum: bool = True


class ChangePasswordRequest(BaseModel):
    username: Optional[str] = None
    old_password: str
    new_password: str


class UpdateAPIKeyRequest(BaseModel):
    new_api_key: str


class UpdateDebugConfigRequest(BaseModel):
    enabled: bool


class UpdateAdminConfigRequest(BaseModel):
    error_ban_threshold: int


class ST2ATRequest(BaseModel):
    """ST转AT请求"""
    st: str


class ImportTokenItem(BaseModel):
    """导入Token项"""
    email: Optional[str] = None
    name: Optional[str] = None
    access_token: Optional[str] = None
    at_expires: Optional[str] = None
    session_token: Optional[str] = None
    cookie: Optional[str] = None
    cookie_file: Optional[str] = None
    is_active: bool = True
    captcha_proxy_url: Optional[str] = None
    image_enabled: bool = True
    video_enabled: bool = True
    image_concurrency: int = -1
    video_concurrency: int = -1


class ImportTokensRequest(BaseModel):
    """导入Token请求"""
    tokens: List[ImportTokenItem]
    confirm_replace_by_email: bool = False


class InternalTokenExportRequest(BaseModel):
    """内部Token导出请求"""
    files: List[str]


class TokenExportRequest(BaseModel):
    """Token 列表导出请求"""
    count: int = Field(default=5, ge=1, le=200)
    node_id: Optional[str] = None
    node_name: Optional[str] = None


# ========== Auth Middleware ==========

async def verify_admin_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(admin_bearer),
):
    """Verify admin session token (NOT API key)"""
    if credentials is None or str(credentials.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing authorization")
    token = str(credentials.credentials or "").strip()

    # Check if token is in active session tokens
    if token not in active_admin_tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired admin token")

    return token


async def verify_cluster_internal_token(
    x_cluster_key: Optional[str] = Header(None, alias="X-Cluster-Key"),
):
    """Verify internal cluster calls between master and workers."""
    if not cluster_manager or not cluster_manager.is_enabled():
        raise HTTPException(status_code=403, detail="Cluster mode is disabled")
    if not cluster_manager.verify_cluster_key(x_cluster_key):
        raise HTTPException(status_code=401, detail="Invalid cluster key")
    return True


# ========== Auth Endpoints ==========

@router.post(
    "/api/admin/login",
    tags=["Admin Auth"],
    summary="管理员登录",
    description="验证后台账号密码并返回后台 session token。该 token 仅用于管理接口，不等于主 API Key。",
)
async def admin_login(request: LoginRequest):
    """Admin login - returns session token (NOT API key)"""
    admin_config = await db.get_admin_config()

    if not AuthManager.verify_admin(request.username, request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate independent session token
    session_token = f"admin-{secrets.token_urlsafe(32)}"

    # Store in active tokens
    active_admin_tokens.add(session_token)

    return {
        "success": True,
        "token": session_token,  # Session token (NOT API key)
        "username": admin_config.username
    }


@router.post(
    "/api/admin/logout",
    tags=["Admin Auth"],
    summary="管理员登出",
    description="使当前后台 session token 失效。",
)
async def admin_logout(token: str = Depends(verify_admin_token)):
    """Admin logout - invalidate session token"""
    active_admin_tokens.discard(token)
    return {"success": True, "message": "退出登录成功"}


@router.post(
    "/api/admin/change-password",
    tags=["Admin Auth"],
    summary="修改管理员密码",
    description="修改后台密码，可选同时修改后台用户名。成功后会清空全部后台 session token，需重新登录。",
)
async def change_password(
    request: ChangePasswordRequest,
    token: str = Depends(verify_admin_token)
):
    """Change admin password"""
    admin_config = await db.get_admin_config()

    # Verify old password
    if not AuthManager.verify_admin(admin_config.username, request.old_password):
        raise HTTPException(status_code=400, detail="旧密码错误")

    # Update password and username in database
    update_params = {"password": request.new_password}
    if request.username:
        update_params["username"] = request.username

    await db.update_admin_config(**update_params)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()

    # 🔑 Invalidate all admin session tokens (force re-login for security)
    active_admin_tokens.clear()

    return {"success": True, "message": "密码修改成功,请重新登录"}


# ========== Token Management ==========

@router.get("/api/tokens")
async def get_tokens(token: str = Depends(verify_admin_token)):
    """Get all tokens with statistics"""
    _ = token
    token_rows = await db.get_all_tokens_with_stats()
    admin_config = await db.get_admin_config()
    error_ban_threshold = int(getattr(admin_config, "error_ban_threshold", 0) or 0)
    to_iso = lambda value: value.isoformat() if hasattr(value, "isoformat") else value

    result = []
    for row in token_rows:
        token_id = int(row.get("id") or 0)
        pending_status = str(row.get("last_refresh_status") or "").strip().upper()
        if pending_status in {"PENDING", "RUNNING", "IN_PROGRESS"}:
            try:
                reconciled = await token_manager.reconcile_pending_auto_login_status(token_id)
                if reconciled is not None:
                    row.update({
                        "st": reconciled.st,
                        "cookie": reconciled.cookie,
                        "cookie_file": reconciled.cookie_file,
                        "at": reconciled.at,
                        "at_expires": reconciled.at_expires,
                        "last_refresh_at": reconciled.last_refresh_at,
                        "last_refresh_method": reconciled.last_refresh_method,
                        "last_refresh_status": reconciled.last_refresh_status,
                        "last_refresh_detail": reconciled.last_refresh_detail,
                        "email": reconciled.email,
                        "name": reconciled.name,
                        "remark": reconciled.remark,
                        "is_active": reconciled.is_active,
                        "ban_reason": reconciled.ban_reason,
                        "banned_at": reconciled.banned_at,
                        "created_at": reconciled.created_at,
                        "last_used_at": reconciled.last_used_at,
                        "use_count": reconciled.use_count,
                        "credits": reconciled.credits,
                        "user_paygate_tier": reconciled.user_paygate_tier,
                        "current_project_id": reconciled.current_project_id,
                        "current_project_name": reconciled.current_project_name,
                        "image_enabled": reconciled.image_enabled,
                        "video_enabled": reconciled.video_enabled,
                        "image_concurrency": reconciled.image_concurrency,
                        "video_concurrency": reconciled.video_concurrency,
                        "captcha_proxy_url": reconciled.captcha_proxy_url,
                    })
            except Exception:
                pass

        disable_reason_meta = _build_disable_reason_meta(row, error_ban_threshold)

        result.append({
            "id": row.get("id"),
            "st": row.get("st"),  # Session Token for editing
            "cookie": row.get("cookie"),  # 完整 Cookie Header（用于 reAuth）
            "cookieFile": row.get("cookie_file"),  # Google 域名 Cookie Header（用于 reAuth step4）
            "at": row.get("at"),  # Access Token for editing (从ST转换而来)
            "at_expires": to_iso(row.get("at_expires")) if row.get("at_expires") else None,
            "last_refresh_at": to_iso(row.get("last_refresh_at")) if row.get("last_refresh_at") else None,
            "last_refresh_method": row.get("last_refresh_method"),
            "last_refresh_status": row.get("last_refresh_status"),
            "last_refresh_detail": row.get("last_refresh_detail"),
            "token": row.get("at"),  # 兼容前端 token.token 的访问方式
            "email": row.get("email"),
            "name": row.get("name"),
            "remark": row.get("remark"),
            "is_active": bool(row.get("is_active")),
            "ban_reason": row.get("ban_reason"),
            "banned_at": to_iso(row.get("banned_at")) if row.get("banned_at") else None,
            "created_at": to_iso(row.get("created_at")) if row.get("created_at") else None,
            "last_used_at": to_iso(row.get("last_used_at")) if row.get("last_used_at") else None,
            "use_count": row.get("use_count"),
            "credits": row.get("credits"),
            "user_paygate_tier": row.get("user_paygate_tier"),
            "current_project_id": row.get("current_project_id"),
            "current_project_name": row.get("current_project_name"),
            "captcha_proxy_url": row.get("captcha_proxy_url") or "",
            "image_enabled": bool(row.get("image_enabled")),
            "video_enabled": bool(row.get("video_enabled")),
            "image_concurrency": row.get("image_concurrency"),
            "video_concurrency": row.get("video_concurrency"),
            "image_count": row.get("image_count", 0),
            "video_count": row.get("video_count", 0),
            "error_count": row.get("error_count", 0),
            "today_error_count": row.get("today_error_count", 0),
            "consecutive_error_count": row.get("consecutive_error_count", 0),
            "last_error_at": to_iso(row.get("last_error_at")) if row.get("last_error_at") else None,
            **disable_reason_meta,
        })

    return result  # 直接返回数组,兼容前端


@router.get("/api/tokens/{token_id}/refresh-history")
async def get_token_refresh_history(
    token_id: int,
    limit: int = Query(100, ge=1, le=500),
    token: str = Depends(verify_admin_token),
):
    """获取某个 Token 的刷新历史记录。"""
    _ = token
    target = await db.get_token(token_id)
    if not target:
        raise HTTPException(status_code=404, detail="token not found")

    rows = await db.get_token_refresh_history(token_id, limit=limit)
    to_iso = lambda value: value.isoformat() if hasattr(value, "isoformat") else value
    history = [
        {
            "id": row.get("id"),
            "token_id": row.get("token_id"),
            "method": row.get("method"),
            "status": row.get("status"),
            "detail": row.get("detail"),
            "created_at": to_iso(row.get("created_at")) if row.get("created_at") else None,
        }
        for row in rows
    ]
    return {
        "success": True,
        "token_id": token_id,
        "email": getattr(target, "email", None),
        "count": len(history),
        "history": history,
    }


@router.get("/api/tokens/project-actions/export")
async def export_code_service_project_actions(token: str = Depends(verify_admin_token)):
    """导出代码服务可直接消费的 project_id/action 列表。"""
    _ = token
    token_rows = await db.get_all_tokens_with_stats()
    items = _build_code_service_project_items(token_rows)
    return {
        "replace_existing": True,
        "force_refill": True,
        "items": items,
    }


def _normalize_internal_token_email_from_name(file_name: str) -> str:
    """从文件名推断邮箱标识：foo_at_bar.com_123456.json -> foo@bar.com"""
    stem = Path(file_name).stem
    stem = re.sub(r"_\d{6,}$", "", stem)
    return stem.replace("_at_", "@")


def _serialize_token_export_item(
    row: Dict[str, Any],
    *,
    exported_node_id: Optional[str] = None,
    exported_node_name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "email": row.get("email"),
        "access_token": row.get("at"),
        "session_token": row.get("st") or None,
        "cookie": row.get("cookie") or None,
        "cookie_file": row.get("cookie_file") or None,
        "is_active": bool(row.get("is_active")),
        "image_enabled": bool(row.get("image_enabled", True)),
        "video_enabled": bool(row.get("video_enabled", True)),
        "image_concurrency": row.get("image_concurrency") if row.get("image_concurrency") is not None else -1,
        "video_concurrency": row.get("video_concurrency") if row.get("video_concurrency") is not None else -1,
        "remark": row.get("remark"),
        "captcha_proxy_url": row.get("captcha_proxy_url") or None,
        "project_id": row.get("current_project_id") or None,
        "project_name": row.get("current_project_name") or None,
        "exported_node_id": exported_node_id,
        "exported_node_name": exported_node_name,
    }


async def _list_export_worker_nodes() -> List[Dict[str, Any]]:
    if not cluster_manager:
        return []

    try:
        snapshot = await cluster_manager.get_cluster_snapshot()
    except Exception:
        return []

    local_node_id = str(snapshot.get("node_id") or "").strip()
    worker_nodes: List[Dict[str, Any]] = []
    for node in snapshot.get("nodes") or []:
        node_id = str(node.get("node_id") or "").strip()
        if not node_id or node_id == local_node_id:
            continue
        if str(node.get("role") or "").strip().lower() != "worker":
            continue
        worker_nodes.append(
            {
                "node_id": node_id,
                "node_name": str(node.get("node_name") or node_id).strip(),
                "base_url": str(node.get("base_url") or "").strip(),
                "healthy": bool(node.get("healthy")),
                "active_requests": int(node.get("active_requests") or 0),
                "available_slots": node.get("available_slots"),
                "weight": int(node.get("weight") or 100),
                "last_seen_at": node.get("last_seen_at"),
            }
        )

    worker_nodes.sort(
        key=lambda item: (
            not bool(item.get("healthy")),
            str(item.get("node_name") or "").lower(),
            str(item.get("node_id") or "").lower(),
        )
    )
    return worker_nodes


@router.get("/api/tokens/export/options")
async def get_token_export_options(token: str = Depends(verify_admin_token)):
    """获取 Token 导出所需的配置和历史信息。"""
    _ = token
    token_rows = await db.get_all_tokens_with_stats()
    active_rows = [row for row in token_rows if bool(row.get("is_active"))]
    bindings = await db.get_token_export_binding_map()
    active_token_ids = {int(row.get("id") or 0) for row in active_rows if int(row.get("id") or 0) > 0}

    binding_counts: Dict[str, int] = {}
    active_bound_count = 0
    for token_id, binding in bindings.items():
        if int(token_id or 0) not in active_token_ids:
            continue
        bound_node_id = str(binding.get("node_id") or "").strip()
        if not bound_node_id:
            continue
        active_bound_count += 1
        binding_counts[bound_node_id] = binding_counts.get(bound_node_id, 0) + 1

    worker_nodes = await _list_export_worker_nodes()
    for index, node in enumerate(worker_nodes):
        node["bound_token_count"] = int(binding_counts.get(str(node.get("node_id") or "").strip(), 0))
        node["default_selected"] = index == 0

    history_rows = await db.get_token_export_history(limit=10)
    history = []
    for row in history_rows:
        created_at = row.get("created_at")
        history.append(
            {
                "id": row.get("id"),
                "node_id": row.get("node_id"),
                "node_name": row.get("node_name"),
                "requested_count": int(row.get("requested_count") or 0),
                "exported_count": int(row.get("exported_count") or 0),
                "token_emails": row.get("token_emails") or [],
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
            }
        )

    return {
        "success": True,
        "default_count": 5,
        "active_token_count": len(active_rows),
        "bound_token_count": active_bound_count,
        "unbound_token_count": max(0, len(active_rows) - active_bound_count),
        "worker_nodes": worker_nodes,
        "history": history,
    }


@router.post("/api/tokens/export")
async def export_tokens_selected(
    request: TokenExportRequest,
    token: str = Depends(verify_admin_token),
):
    """按数量导出 Token，并在需要时绑定到指定子节点。"""
    _ = token
    requested_count = max(1, int(request.count or 5))
    node_id = str(request.node_id or "").strip() or None
    node_name = str(request.node_name or "").strip() or None

    worker_nodes = await _list_export_worker_nodes()
    worker_node_map = {
        str(node.get("node_id") or "").strip(): node
        for node in worker_nodes
        if str(node.get("node_id") or "").strip()
    }
    if node_id and node_id not in worker_node_map:
        raise HTTPException(status_code=400, detail="所选子节点不存在或当前不可用")
    if node_id and node_id in worker_node_map:
        node_name = str(worker_node_map[node_id].get("node_name") or node_name or node_id).strip() or node_id

    token_rows = await db.get_all_tokens_with_stats()
    active_rows = [row for row in token_rows if bool(row.get("is_active"))]
    if not active_rows:
        raise HTTPException(status_code=404, detail="当前没有可导出的活跃 Token")

    bindings = await db.get_token_export_binding_map()
    selected_rows: List[Dict[str, Any]] = []
    if node_id:
        same_node_rows: List[Dict[str, Any]] = []
        unbound_rows: List[Dict[str, Any]] = []
        for row in active_rows:
            token_id = int(row.get("id") or 0)
            binding = bindings.get(token_id) or {}
            binding_node_id = str(binding.get("node_id") or "").strip()
            if binding_node_id == node_id:
                same_node_rows.append(row)
            elif not binding_node_id:
                unbound_rows.append(row)

        same_node_rows.sort(
            key=lambda row: (
                str((bindings.get(int(row.get("id") or 0)) or {}).get("last_exported_at") or ""),
                str(row.get("created_at") or ""),
            ),
            reverse=True,
        )
        selected_rows = (same_node_rows + unbound_rows)[:requested_count]
    else:
        selected_rows = active_rows[:requested_count]

    if not selected_rows:
        if node_id:
            raise HTTPException(status_code=404, detail="该子节点当前没有可分配的活跃 Token")
        raise HTTPException(status_code=404, detail="当前没有可导出的 Token")

    exported_items = [
        _serialize_token_export_item(
            row,
            exported_node_id=node_id,
            exported_node_name=node_name,
        )
        for row in selected_rows
    ]
    exported_count = len(exported_items)
    payload = json.dumps(exported_items, ensure_ascii=False, indent=2)
    payload_buffer = io.BytesIO(payload.encode("utf-8"))
    payload_buffer.seek(0)

    exported_token_ids = [int(row.get("id") or 0) for row in selected_rows if int(row.get("id") or 0) > 0]
    exported_token_emails = [str(row.get("email") or "").strip() for row in selected_rows if str(row.get("email") or "").strip()]
    export_time = datetime.utcnow()

    if node_id and exported_token_ids:
        await db.upsert_token_export_bindings(
            exported_token_ids,
            node_id=node_id,
            node_name=node_name,
            exported_at=export_time,
        )

    await db.add_token_export_history(
        node_id=node_id,
        node_name=node_name,
        requested_count=requested_count,
        exported_count=exported_count,
        token_ids=exported_token_ids,
        token_emails=exported_token_emails,
        created_at=export_time,
    )

    filename_parts = ["tokens", datetime.now().strftime("%Y%m%d_%H%M%S")]
    if node_id:
        safe_node = re.sub(r"[^a-zA-Z0-9._-]+", "_", node_id).strip("_") or "node"
        filename_parts.append(safe_node)
    filename = "_".join(filename_parts) + ".json"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Exported-Count": str(exported_count),
        "X-Requested-Count": str(requested_count),
    }
    return StreamingResponse(payload_buffer, media_type="application/json", headers=headers)


def _scan_internal_token_files(token_dir: Path) -> List[Dict[str, Any]]:
    """扫描 tmp/Token 下 JSON 文件并标记每个邮箱的最新文件。"""
    json_files = sorted(token_dir.glob("*.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="tmp/Token 下没有可导出的 JSON 文件")

    file_items: List[Dict[str, Any]] = []
    latest_by_email: Dict[str, Dict[str, Any]] = {}

    for file_path in json_files:
        stat = file_path.stat()
        email = _normalize_internal_token_email_from_name(file_path.name)
        item = {
            "name": file_path.name,
            "email": email,
            "size": int(stat.st_size),
            "mtime_ts": float(stat.st_mtime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_latest_for_email": False,
        }
        file_items.append(item)

        current_latest = latest_by_email.get(email)
        if (
            current_latest is None
            or item["mtime_ts"] > current_latest["mtime_ts"]
            or (
                item["mtime_ts"] == current_latest["mtime_ts"]
                and str(item["name"]) > str(current_latest["name"])
            )
        ):
            latest_by_email[email] = item

    latest_names = {v["name"] for v in latest_by_email.values()}
    for item in file_items:
        item["is_latest_for_email"] = item["name"] in latest_names
        item["default_selected"] = item["is_latest_for_email"]

    # 展示顺序：邮箱升序，邮箱内按修改时间降序
    file_items.sort(key=lambda x: (str(x["email"]).lower(), -float(x["mtime_ts"]), str(x["name"]).lower()))
    return file_items


def _build_internal_export_zip_response(
    token_dir: Path,
    selected_names: List[str],
    filename_prefix: str = "internal_tokens",
) -> StreamingResponse:
    """按指定文件名打包导出 zip。"""
    available_map = {p.name: p for p in token_dir.glob("*.json")}
    normalized_selected: List[str] = []
    seen = set()
    for name in selected_names:
        n = str(name or "").strip()
        if not n or n in seen:
            continue
        normalized_selected.append(n)
        seen.add(n)

    if not normalized_selected:
        raise HTTPException(status_code=400, detail="未选择任何可导出的 JSON 文件")

    invalid_names = [n for n in normalized_selected if n not in available_map]
    if invalid_names:
        raise HTTPException(
            status_code=400,
            detail=f"存在无效文件: {', '.join(invalid_names[:5])}",
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in normalized_selected:
            file_path = available_map[file_name]
            zf.write(file_path, arcname=file_path.name)
    zip_buffer.seek(0)

    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@router.get("/api/tokens/internal/files")
async def list_internal_token_files(token: str = Depends(verify_admin_token)):
    """列出 tmp/Token 下可导出的 JSON 文件，并标记每个邮箱最新文件。"""
    _ = token
    repo_root = Path(__file__).resolve().parents[2]
    token_dir = repo_root / "tmp" / "Token"
    if not token_dir.exists() or not token_dir.is_dir():
        file_items = []
    else:
        try:
            file_items = _scan_internal_token_files(token_dir)
        except HTTPException as e:
            if e.status_code == 404:
                file_items = []
            else:
                raise

    default_selected_names = [f["name"] for f in file_items if f.get("default_selected")]
    unique_emails = len({str(f.get("email") or "").strip().lower() for f in file_items})

    return {
        "success": True,
        "token_dir": "tmp/Token",
        "total_files": len(file_items),
        "unique_emails": unique_emails,
        "default_selected_count": len(default_selected_names),
        "files": file_items,
    }


@router.get("/api/tokens/internal/export")
async def export_internal_tokens(token: str = Depends(verify_admin_token)):
    """导出 tmp/Token 去重后的默认文件（每个邮箱仅最新文件）。"""
    _ = token
    repo_root = Path(__file__).resolve().parents[2]
    token_dir = repo_root / "tmp" / "Token"
    if not token_dir.exists() or not token_dir.is_dir():
        raise HTTPException(status_code=404, detail="tmp/Token 目录不存在")

    file_items = _scan_internal_token_files(token_dir)
    default_selected_names = [f["name"] for f in file_items if f.get("default_selected")]
    return _build_internal_export_zip_response(
        token_dir=token_dir,
        selected_names=default_selected_names,
        filename_prefix="internal_tokens",
    )


@router.post("/api/tokens/internal/export")
async def export_internal_tokens_selected(
    request: InternalTokenExportRequest,
    token: str = Depends(verify_admin_token),
):
    """按勾选文件导出 tmp/Token JSON 压缩包。"""
    _ = token
    repo_root = Path(__file__).resolve().parents[2]
    token_dir = repo_root / "tmp" / "Token"
    if not token_dir.exists() or not token_dir.is_dir():
        raise HTTPException(status_code=404, detail="tmp/Token 目录不存在")

    return _build_internal_export_zip_response(
        token_dir=token_dir,
        selected_names=request.files or [],
        filename_prefix="internal_tokens_selected",
    )


@router.post("/api/tokens")
async def add_token(
    request: AddTokenRequest,
    token: str = Depends(verify_admin_token)
):
    """Add a new token"""
    try:
        new_token = await token_manager.add_token(
            st=request.st,
            cookie=request.cookie,
            cookie_file=request.cookie_file,
            project_id=request.project_id,  # 🆕 支持用户指定project_id
            project_name=request.project_name,
            remark=request.remark,
            captcha_proxy_url=request.captcha_proxy_url.strip() if request.captcha_proxy_url is not None else None,
            image_enabled=request.image_enabled,
            video_enabled=request.video_enabled,
            image_concurrency=request.image_concurrency,
            video_concurrency=request.video_concurrency
        )

        # 热更新并发限制，避免必须重启服务
        if concurrency_manager:
            await concurrency_manager.reset_token(
                new_token.id,
                image_concurrency=new_token.image_concurrency,
                video_concurrency=new_token.video_concurrency
            )

        return {
            "success": True,
            "message": "Token添加成功",
            "token": {
                "id": new_token.id,
                "email": new_token.email,
                "credits": new_token.credits,
                "project_id": new_token.current_project_id,
                "project_name": new_token.current_project_name
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加Token失败: {str(e)}")


@router.put("/api/tokens/{token_id}")
async def update_token(
    token_id: int,
    request: UpdateTokenRequest,
    token: str = Depends(verify_admin_token)
):
    """Update token - 使用ST自动刷新AT"""
    try:
        # 先ST转AT
        result = await token_manager.flow_client.st_to_at(request.st)
        at = result["access_token"]
        expires = result.get("expires")

        # 解析过期时间
        from datetime import datetime
        at_expires = None
        if expires:
            try:
                at_expires = datetime.fromisoformat(expires.replace('Z', '+00:00'))
            except:
                pass

        # 更新token (包含AT、ST、AT过期时间、project_id和project_name)
        await token_manager.update_token(
            token_id=token_id,
            st=request.st,
            cookie=request.cookie,
            cookie_file=request.cookie_file,
            at=at,
            at_expires=at_expires,  # 🆕 更新AT过期时间
            project_id=request.project_id,
            project_name=request.project_name,
            remark=request.remark,
            captcha_proxy_url=request.captcha_proxy_url.strip() if request.captcha_proxy_url is not None else None,
            image_enabled=request.image_enabled,
            video_enabled=request.video_enabled,
            image_concurrency=request.image_concurrency,
            video_concurrency=request.video_concurrency
        )

        # 热更新并发限制，确保管理台修改立即生效
        if concurrency_manager:
            updated_token = await token_manager.get_token(token_id)
            if updated_token:
                await concurrency_manager.reset_token(
                    token_id,
                    image_concurrency=updated_token.image_concurrency,
                    video_concurrency=updated_token.video_concurrency
                )

        return {"success": True, "message": "Token更新成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/tokens/{token_id}")
async def delete_token(
    token_id: int,
    token: str = Depends(verify_admin_token)
):
    """Delete token"""
    try:
        await token_manager.delete_token(token_id)
        if concurrency_manager:
            await concurrency_manager.remove_token(token_id)
        return {"success": True, "message": "Token删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/tokens")
async def delete_all_tokens(token: str = Depends(verify_admin_token)):
    """Delete all tokens"""
    try:
        deleted = await token_manager.delete_all_tokens()
        if concurrency_manager:
            await concurrency_manager.initialize([])
        return {
            "success": True,
            "message": f"已删除全部Token（{deleted}个）",
            "deleted": deleted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/tokens/{token_id}/enable")
async def enable_token(
    token_id: int,
    token: str = Depends(verify_admin_token)
):
    """Enable token"""
    await token_manager.enable_token(token_id)
    return {"success": True, "message": "Token已启用"}


@router.post("/api/tokens/{token_id}/disable")
async def disable_token(
    token_id: int,
    token: str = Depends(verify_admin_token)
):
    """Disable token"""
    await token_manager.disable_token(token_id, reason="manual_disabled")
    return {"success": True, "message": "Token已禁用"}


@router.post("/api/tokens/{token_id}/refresh-credits")
async def refresh_credits(
    token_id: int,
    token: str = Depends(verify_admin_token)
):
    """刷新Token余额 🆕"""
    try:
        credits = await token_manager.refresh_credits(token_id)
        return {
            "success": True,
            "message": "余额刷新成功",
            "credits": credits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新余额失败: {str(e)}")


@router.post("/api/tokens/{token_id}/refresh-at")
async def refresh_at(
    token_id: int,
    token: str = Depends(verify_admin_token)
):
    """手动刷新Token的AT (使用ST转换) 🆕
    
    如果 AT 刷新失败且处于 personal 模式，会自动尝试通过浏览器刷新 ST
    """
    from ..core.logger import debug_logger
    from ..core.config import config
    
    debug_logger.log_info(f"[API] 手动刷新 AT 请求: token_id={token_id}, captcha_method={config.captcha_method}")
    
    try:
        # 调用token_manager的内部刷新方法（包含 ST 自动刷新逻辑）
        success = await token_manager._refresh_at(token_id, refresh_source="MANUAL_AT")

        if success:
            # 获取更新后的token信息
            updated_token = await token_manager.get_token(token_id)
            
            message = "AT刷新成功"
            if config.captcha_method == "personal":
                message += "（支持ST自动刷新）"
            
            debug_logger.log_info(f"[API] AT 刷新成功: token_id={token_id}")
            
            return {
                "success": True,
                "message": message,
                "token": {
                    "id": updated_token.id,
                    "email": updated_token.email,
                    "at_expires": updated_token.at_expires.isoformat() if updated_token.at_expires else None,
                    "last_refresh_at": updated_token.last_refresh_at.isoformat() if updated_token.last_refresh_at else None,
                    "last_refresh_method": updated_token.last_refresh_method,
                    "last_refresh_status": updated_token.last_refresh_status,
                    "last_refresh_detail": updated_token.last_refresh_detail,
                }
            }
        else:
            debug_logger.log_error(f"[API] AT 刷新失败: token_id={token_id}")
            
            error_detail = "AT刷新失败"
            if config.captcha_method != "personal":
                error_detail += f"（当前打码模式: {config.captcha_method}，ST自动刷新仅在 personal 模式下可用）"
            
            raise HTTPException(status_code=500, detail=error_detail)
    except HTTPException:
        raise
    except Exception as e:
        debug_logger.log_error(f"[API] 刷新AT异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刷新AT失败: {str(e)}")


@router.post("/api/tokens/{token_id}/refresh-cookie")
async def refresh_cookie(
    token_id: int,
    token: str = Depends(verify_admin_token)
):
    """手动仅通过 reAuth 刷新 Cookie（跳过首次 ST->AT 直刷）"""
    from ..core.logger import debug_logger

    debug_logger.log_info(f"[API] 手动刷新 Cookie(reAuth-only) 请求: token_id={token_id}")

    try:
        success = await token_manager.refresh_cookie_via_reauth(token_id, refresh_source="MANUAL_COOKIE")
        if success:
            updated_token = await token_manager.get_token(token_id)
            debug_logger.log_info(f"[API] 刷新 Cookie 成功: token_id={token_id}")
            return {
                "success": True,
                "message": "Cookie刷新成功（reAuth-only）",
                "token": {
                    "id": updated_token.id,
                    "email": updated_token.email,
                    "at_expires": updated_token.at_expires.isoformat() if updated_token.at_expires else None,
                    "cookie_present": bool(str(updated_token.cookie or "").strip()),
                    "last_refresh_at": updated_token.last_refresh_at.isoformat() if updated_token.last_refresh_at else None,
                    "last_refresh_method": updated_token.last_refresh_method,
                    "last_refresh_status": updated_token.last_refresh_status,
                    "last_refresh_detail": updated_token.last_refresh_detail,
                }
            }

        updated_token = await token_manager.get_token(token_id)
        fail_detail = (
            str(getattr(updated_token, "last_refresh_detail", "") or "").strip()
            or "Cookie刷新失败（reAuth-only）"
        )
        debug_logger.log_error(f"[API] 刷新 Cookie 失败: token_id={token_id}, detail={fail_detail}")
        raise HTTPException(status_code=500, detail=fail_detail)
    except HTTPException:
        raise
    except Exception as e:
        debug_logger.log_error(f"[API] 刷新Cookie异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刷新Cookie失败: {str(e)}")


@router.post("/api/tokens/st2at")
async def st_to_at(
    request: ST2ATRequest,
    token: str = Depends(verify_admin_token)
):
    """Convert Session Token to Access Token (仅转换,不添加到数据库)"""
    try:
        result = await token_manager.flow_client.st_to_at(request.st)
        return {
            "success": True,
            "message": "ST converted to AT successfully",
            "access_token": result["access_token"],
            "email": result.get("user", {}).get("email"),
            "expires": result.get("expires")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/tokens/import")
async def import_tokens(
    request: ImportTokensRequest,
    token: str = Depends(verify_admin_token)
):
    """批量导入Token"""
    from datetime import datetime, timezone

    added = 0
    updated = 0
    skipped = 0
    errors = []
    def _normalize_email(value: Optional[str]) -> str:
        return str(value or "").strip().lower()

    def _parse_expires(value: Optional[str], fallback_token: Optional[str] = None) -> Optional[datetime]:
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except Exception:
                pass
        if fallback_token:
            try:
                return token_manager._parse_jwt_exp(fallback_token)
            except Exception:
                return None
        return None

    def _cache_existing_token(email_value: Optional[str], token_obj: Any):
        raw_email = str(email_value or "").strip()
        if raw_email and raw_email not in existing_by_email:
            existing_by_email[raw_email] = token_obj
        email_key = _normalize_email(email_value)
        if email_key:
            existing_by_email[email_key] = token_obj

    # 保持与历史逻辑一致：按 created_at DESC 的结果中，优先命中同邮箱“最新一条”
    existing_by_email: Dict[str, Any] = {}
    existing_tokens = await token_manager.get_all_tokens()
    for existing_token in existing_tokens:
        _cache_existing_token(existing_token.email, existing_token)

    semaphore = asyncio.Semaphore(6)

    async def _resolve_import_item(idx: int, item: ImportTokenItem) -> Dict[str, Any]:
        st = str(item.session_token or "").strip()
        if not st:
            return {"error": f"第{idx+1}项: 缺少 session_token"}

        email_hint_raw = str(item.email or "").strip()
        name_hint = str(item.name or "").strip()

        try:
            async with semaphore:
                result = await token_manager.flow_client.st_to_at(st)
            access_token_hint = str(result.get("access_token") or "").strip()
            email_hint_raw = str(result.get("user", {}).get("email") or email_hint_raw or "").strip()
            email_hint = _normalize_email(email_hint_raw)
            name_hint = str(result.get("user", {}).get("name") or name_hint or "").strip()
            at_expires = _parse_expires(result.get("expires"), fallback_token=access_token_hint)

            if not email_hint:
                return {"error": f"第{idx+1}项: 无法获取邮箱信息"}

            now = datetime.now(timezone.utc)
            return {
                "idx": idx,
                "item": item,
                "st": st,
                "email": email_hint,
                "name": name_hint,
                "at": access_token_hint or None,
                "at_expires": at_expires,
                "is_expired": bool(at_expires and at_expires <= now),
            }
        except Exception as e:
            return {"error": f"第{idx+1}项: {str(e)}"}

    resolved_items = await asyncio.gather(
        *(_resolve_import_item(idx, item) for idx, item in enumerate(request.tokens))
    )

    for resolved in resolved_items:
        if resolved.get("error"):
            errors.append(resolved["error"])
            continue

        try:
            item = resolved["item"]
            st = resolved["st"]
            email = resolved["email"]
            at = resolved.get("at")
            at_expires = resolved.get("at_expires")
            is_expired = bool(resolved.get("is_expired"))

            existing = existing_by_email.get(email)
            if existing:
                if not request.confirm_replace_by_email:
                    skipped += 1
                    continue

                await token_manager.update_token(
                    token_id=existing.id,
                    st=st,
                    cookie=item.cookie,
                    cookie_file=item.cookie_file,
                    at=at,
                    at_expires=at_expires,
                    captcha_proxy_url=item.captcha_proxy_url.strip() if item.captcha_proxy_url is not None else None,
                    image_enabled=item.image_enabled,
                    video_enabled=item.video_enabled,
                    image_concurrency=item.image_concurrency,
                    video_concurrency=item.video_concurrency
                )
                if item.is_active:
                    await token_manager.db.update_token(
                        existing.id,
                        is_active=True,
                        ban_reason=None,
                        banned_at=None
                    )
                else:
                    await token_manager.disable_token(existing.id, reason="import_disabled")

                if is_expired:
                    await token_manager.disable_token(existing.id, reason="token_expired")
                    existing.is_active = False

                existing.st = st
                if at is not None:
                    existing.at = at
                existing.at_expires = at_expires
                existing.cookie = item.cookie
                existing.cookie_file = item.cookie_file
                existing.captcha_proxy_url = item.captcha_proxy_url
                existing.image_enabled = item.image_enabled
                existing.video_enabled = item.video_enabled
                existing.image_concurrency = item.image_concurrency
                existing.video_concurrency = item.video_concurrency
                _cache_existing_token(email, existing)
                updated += 1
            else:
                new_token = await token_manager.add_token(
                    st=st,
                    captcha_proxy_url=item.captcha_proxy_url.strip() if item.captcha_proxy_url is not None else None,
                    cookie=item.cookie,
                    cookie_file=item.cookie_file,
                    image_enabled=item.image_enabled,
                    video_enabled=item.video_enabled,
                    image_concurrency=item.image_concurrency,
                    video_concurrency=item.video_concurrency,
                    resolved_at=at,
                    resolved_at_expires=at_expires,
                    resolved_email=email,
                    resolved_name=resolved.get("name"),
                )
                if not item.is_active:
                    await token_manager.disable_token(new_token.id, reason="import_disabled")
                if is_expired:
                    await token_manager.disable_token(new_token.id, reason="token_expired")
                    new_token.is_active = False

                _cache_existing_token(email, new_token)
                added += 1
        except Exception as e:
            idx = int(resolved.get("idx", 0))
            errors.append(f"第{idx+1}项: {str(e)}")

    return {
        "success": True,
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "errors": errors if errors else None,
        "message": (
            f"导入完成: 新增 {added} 个, 更新 {updated} 个, 跳过 {skipped} 个"
            + (f", {len(errors)} 个失败" if errors else "")
        )
    }


# ========== Config Management ==========

@router.get("/api/server/config")
async def get_server_config(token: str = Depends(verify_admin_token)):
    """Get server runtime mode and bind config from setting.toml."""
    return {
        "success": True,
        "config": {
            "mode": config.get_server_mode(),
            "host": config.server_host,
            "configured_host": config.configured_server_host,
            "port": config.server_port,
            "default_public_ip": config.default_server_public_ip,
            "detected_public_ip": config.detected_public_ip,
            "server_auto_detected": config.server_auto_detected,
            "rpa_test_bitbrowser_id_local": config.get_rpa_test_bitbrowser_id_local(),
            "rpa_test_bitbrowser_id_server": config.get_rpa_test_bitbrowser_id_server(),
            "rpa_test_bitbrowser_ids_local": config.get_rpa_test_bitbrowser_ids_local(),
            "rpa_test_bitbrowser_ids_server": config.get_rpa_test_bitbrowser_ids_server(),
            "rpa_test_bitbrowser_id_active": config.get_active_rpa_test_bitbrowser_id(),
            "block_gemini_25_flash_image": bool(config.block_gemini_25_flash_image),
            "restart_required": True,
        },
    }


@router.post("/api/server/config")
async def update_server_config(
    request: ServerConfigRequest,
    token: str = Depends(verify_admin_token),
):
    """Update server bind config in setting.toml.

    Note: this only affects next process start; restart is required.
    """
    mode = str(request.mode or "").strip().lower()
    if mode not in {"local", "server"}:
        raise HTTPException(status_code=400, detail="mode 必须是 local 或 server")

    host = str(request.host or "").strip()
    default_public_ip = str(request.default_public_ip or "").strip()
    if not host:
        host = "127.0.0.1" if mode == "local" else "0.0.0.0"
    host = config.normalize_server_host_for_mode(
        mode=mode,
        host=host,
        default_public_ip=default_public_ip,
    )
    port = request.port if request.port is not None else int(config.server_port)
    local_bit_ids = (
        [str(v or "").strip() for v in (request.rpa_test_bitbrowser_ids_local or [])]
        if request.rpa_test_bitbrowser_ids_local is not None
        else (
            [str(request.rpa_test_bitbrowser_id_local).strip()]
            if request.rpa_test_bitbrowser_id_local is not None
            else config.get_rpa_test_bitbrowser_ids_local()
        )
    )
    server_bit_ids = (
        [str(v or "").strip() for v in (request.rpa_test_bitbrowser_ids_server or [])]
        if request.rpa_test_bitbrowser_ids_server is not None
        else (
            [str(request.rpa_test_bitbrowser_id_server).strip()]
            if request.rpa_test_bitbrowser_id_server is not None
            else config.get_rpa_test_bitbrowser_ids_server()
        )
    )
    block_gemini_25_flash_image = (
        bool(request.block_gemini_25_flash_image)
        if request.block_gemini_25_flash_image is not None
        else bool(config.block_gemini_25_flash_image)
    )

    try:
        config.update_server_config(
            host=host,
            port=port,
            default_public_ip=default_public_ip,
        )
        config.update_rpa_test_bitbrowser_ids(
            local_ids=local_bit_ids,
            server_ids=server_bit_ids,
        )
        config.update_flow_switches(
            block_gemini_25_flash_image=block_gemini_25_flash_image,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "message": "系统配置已保存，模型拦截开关立即生效；监听地址/端口修改需重启服务",
        "config": {
            "mode": config.get_server_mode(),
            "host": config.server_host,
            "configured_host": config.configured_server_host,
            "port": config.server_port,
            "default_public_ip": config.default_server_public_ip,
            "detected_public_ip": config.detected_public_ip,
            "server_auto_detected": config.server_auto_detected,
            "rpa_test_bitbrowser_id_local": config.get_rpa_test_bitbrowser_id_local(),
            "rpa_test_bitbrowser_id_server": config.get_rpa_test_bitbrowser_id_server(),
            "rpa_test_bitbrowser_ids_local": config.get_rpa_test_bitbrowser_ids_local(),
            "rpa_test_bitbrowser_ids_server": config.get_rpa_test_bitbrowser_ids_server(),
            "rpa_test_bitbrowser_id_active": config.get_active_rpa_test_bitbrowser_id(),
            "block_gemini_25_flash_image": bool(config.block_gemini_25_flash_image),
            "restart_required": True,
        },
    }


@router.get("/api/cluster/config")
async def get_cluster_config(token: str = Depends(verify_admin_token)):
    """Get cluster role/config from setting.toml."""
    _ = token
    return {
        "success": True,
        "config": {
            "enabled": bool(config.cluster_enabled),
            "role": config.cluster_role,
            "node_id": config.cluster_node_id,
            "node_name": config.cluster_node_name,
            "master_base_url": config.cluster_master_base_url,
            "node_public_base_url": config.cluster_node_public_base_url,
            "effective_node_public_base_url": config.cluster_effective_node_public_base_url,
            "cluster_key": config.cluster_key,
            "cluster_key_masked": config.mask_cluster_key(),
            "node_weight": int(config.cluster_node_weight),
            "node_max_concurrency": int(config.cluster_node_max_concurrency),
            "heartbeat_interval_seconds": int(config.cluster_heartbeat_interval_seconds),
            "heartbeat_timeout_seconds": int(config.cluster_heartbeat_timeout_seconds),
            "dispatch_timeout_seconds": int(config.cluster_dispatch_timeout_seconds),
            "prefer_local": bool(config.cluster_prefer_local),
            "restart_required": False,
        },
    }


@router.post("/api/cluster/config")
async def update_cluster_config_endpoint(
    request: ClusterConfigRequest,
    token: str = Depends(verify_admin_token),
):
    """Update cluster configuration and hot-reload runtime tasks."""
    _ = token
    try:
        config.update_cluster_config(
            role=request.role,
            node_id=request.node_id,
            node_name=request.node_name,
            master_base_url=request.master_base_url,
            node_public_base_url=request.node_public_base_url,
            cluster_key=request.cluster_key,
            node_weight=request.node_weight,
            node_max_concurrency=request.node_max_concurrency,
            heartbeat_interval_seconds=request.heartbeat_interval_seconds,
            heartbeat_timeout_seconds=request.heartbeat_timeout_seconds,
            dispatch_timeout_seconds=request.dispatch_timeout_seconds,
            prefer_local=request.prefer_local,
        )
        if cluster_manager:
            await cluster_manager.reload_runtime()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await get_cluster_config(token)


@router.post("/api/cluster/rotate-key")
async def rotate_cluster_key(token: str = Depends(verify_admin_token)):
    """Rotate shared cluster key used by master/workers."""
    _ = token
    new_key = config.rotate_cluster_key()
    if cluster_manager:
        await cluster_manager.reload_runtime()
    return {
        "success": True,
        "cluster_key": new_key,
        "cluster_key_masked": config.mask_cluster_key(),
        "message": "Cluster Key 已轮换。请同步更新所有子节点配置。",
    }


@router.get("/api/cluster/diagnostics")
async def get_cluster_diagnostics(token: str = Depends(verify_admin_token)):
    """Get current cluster nodes and health snapshot."""
    _ = token
    if not cluster_manager:
        return {
            "success": True,
            "snapshot": {
                "enabled": False,
                "role": "standalone",
                "total_nodes": 0,
                "healthy_nodes": 0,
                "effective_capacity": 0,
                "available_capacity": 0,
                "nodes": [],
            },
        }
    return {"success": True, "snapshot": await cluster_manager.get_cluster_snapshot()}


@router.post("/api/cluster/delete-node")
async def delete_cluster_node(
    request: ClusterDeleteNodeRequest,
    token: str = Depends(verify_admin_token),
):
    """Delete a remote node record from the master's node list."""
    _ = token
    if not cluster_manager or not cluster_manager.is_enabled():
        raise HTTPException(status_code=400, detail="当前未启用集群模式")
    if not cluster_manager.is_master():
        raise HTTPException(status_code=403, detail="只有主节点可以删除节点记录")

    try:
        deleted = await cluster_manager.remove_node(request.node_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "success": True,
        "deleted": bool(deleted),
        "node_id": request.node_id,
        "message": "节点记录已删除" if deleted else "节点记录不存在，可能已被清理",
        "snapshot": await cluster_manager.get_cluster_snapshot(),
    }


@router.post("/api/internal/cluster/heartbeat", include_in_schema=False)
async def receive_cluster_heartbeat(
    request: ClusterHeartbeatRequest,
    authorized: bool = Depends(verify_cluster_internal_token),
):
    """Worker heartbeat endpoint consumed by master node."""
    _ = authorized
    if not cluster_manager or not cluster_manager.is_master():
        raise HTTPException(status_code=403, detail="当前节点不是主节点")
    try:
        return await cluster_manager.register_heartbeat(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/config/proxy")
async def get_proxy_config(token: str = Depends(verify_admin_token)):
    """Get proxy configuration"""
    config = await proxy_manager.get_proxy_config()
    return {
        "success": True,
        "config": {
            "enabled": config.enabled,
            "proxy_url": config.proxy_url,
            "media_proxy_enabled": config.media_proxy_enabled,
            "media_proxy_url": config.media_proxy_url
        }
    }


@router.get("/api/proxy/config")
async def get_proxy_config_alias(token: str = Depends(verify_admin_token)):
    """Get proxy configuration (alias for frontend compatibility)"""
    config = await proxy_manager.get_proxy_config()
    return {
        "proxy_enabled": config.enabled,  # Frontend expects proxy_enabled
        "proxy_url": config.proxy_url,
        "media_proxy_enabled": config.media_proxy_enabled,
        "media_proxy_url": config.media_proxy_url
    }


@router.post("/api/proxy/config")
async def update_proxy_config_alias(
    request: ProxyConfigRequest,
    token: str = Depends(verify_admin_token)
):
    """Update proxy configuration (alias for frontend compatibility)"""
    try:
        await proxy_manager.update_proxy_config(
            enabled=request.proxy_enabled,
            proxy_url=request.proxy_url,
            media_proxy_enabled=request.media_proxy_enabled,
            media_proxy_url=request.media_proxy_url
        )
    except ValueError as e:
        return {"success": False, "message": str(e)}
    return {"success": True, "message": "代理配置更新成功"}


@router.post("/api/config/proxy")
async def update_proxy_config(
    request: ProxyConfigRequest,
    token: str = Depends(verify_admin_token)
):
    """Update proxy configuration"""
    try:
        await proxy_manager.update_proxy_config(
            enabled=request.proxy_enabled,
            proxy_url=request.proxy_url,
            media_proxy_enabled=request.media_proxy_enabled,
            media_proxy_url=request.media_proxy_url
        )
    except ValueError as e:
        return {"success": False, "message": str(e)}
    return {"success": True, "message": "代理配置更新成功"}


@router.post("/api/proxy/test")
async def test_proxy_connectivity(
    request: ProxyTestRequest,
    token: str = Depends(verify_admin_token)
):
    """测试代理是否可访问目标站点（默认 https://labs.google/）"""
    proxy_input = (request.proxy_url or "").strip()
    test_url = (request.test_url or "https://labs.google/").strip()
    timeout_seconds = int(request.timeout_seconds or 15)
    timeout_seconds = max(5, min(timeout_seconds, 60))

    if not proxy_input:
        return {
            "success": False,
            "message": "代理地址为空",
            "test_url": test_url
        }

    try:
        proxy_url = proxy_manager.normalize_proxy_url(proxy_input)
    except ValueError as e:
        return {
            "success": False,
            "message": str(e),
            "test_url": test_url
        }

    start_time = time.time()
    try:
        proxies = {"http": proxy_url, "https": proxy_url}
        async with AsyncSession() as session:
            resp = await session.get(
                test_url,
                proxies=proxies,
                timeout=timeout_seconds,
                impersonate="chrome120",
                allow_redirects=True,
                verify=False
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        status_code = resp.status_code
        final_url = str(resp.url)
        ok = 200 <= status_code < 400

        return {
            "success": ok,
            "message": "代理可用" if ok else f"代理可连通，但目标返回状态码 {status_code}",
            "test_url": test_url,
            "final_url": final_url,
            "status_code": status_code,
            "elapsed_ms": elapsed_ms
        }
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "message": f"代理测试失败: {str(e)}",
            "test_url": test_url,
            "elapsed_ms": elapsed_ms
        }


@router.get("/api/config/generation")
async def get_generation_config(token: str = Depends(verify_admin_token)):
    """Get generation timeout configuration"""
    config = await db.get_generation_config()
    return {
        "success": True,
        "config": {
            "image_timeout": config.image_timeout,
            "video_timeout": config.video_timeout
        }
    }


@router.post("/api/config/generation")
async def update_generation_config(
    request: GenerationConfigRequest,
    token: str = Depends(verify_admin_token)
):
    """Update generation timeout configuration"""
    await db.update_generation_config(request.image_timeout, request.video_timeout)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()

    return {"success": True, "message": "生成配置更新成功"}


# ========== System Info ==========

@router.get("/api/system/info")
async def get_system_info(token: str = Depends(verify_admin_token)):
    """Get system information"""
    stats = await db.get_system_info_stats()

    return {
        "success": True,
        "info": {
            "total_tokens": stats["total_tokens"],
            "active_tokens": stats["active_tokens"],
            "total_credits": stats["total_credits"],
            "version": "1.0.0"
        }
    }


# ========== Additional Routes for Frontend Compatibility ==========

@router.post("/api/login")
async def login(request: LoginRequest):
    """Login endpoint (alias for /api/admin/login)"""
    return await admin_login(request)


@router.post("/api/logout")
async def logout(token: str = Depends(verify_admin_token)):
    """Logout endpoint (alias for /api/admin/logout)"""
    return await admin_logout(token)


@router.get("/api/stats")
async def get_stats(token: str = Depends(verify_admin_token)):
    """Get statistics for dashboard"""
    return await db.get_dashboard_stats()


@router.get("/api/concurrency-status")
async def get_concurrency_status(token: str = Depends(verify_admin_token)):
    """Get real-time concurrency / load status for all tokens"""
    if load_balancer:
        return await load_balancer.get_all_load_status()
    if not concurrency_manager:
        return {"tokens": {}, "summary": {
            "total_image_inflight": 0,
            "total_video_inflight": 0,
            "total_image_capacity": None,
            "total_video_capacity": None,
        }}
    return await concurrency_manager.get_all_concurrency_status()


@router.get("/api/diagnostics")
async def get_diagnostics(token: str = Depends(verify_admin_token)):
    """Get comprehensive server performance diagnostics snapshot.
    
    Returns in-flight requests, duration percentiles, phase breakdown,
    throughput timeline, slow requests, event loop lag, and system resource usage.
    """
    return await perf_monitor.get_diagnostics()


@router.get("/api/diagnostics/history")
async def get_diagnostics_history(
    minutes: int = Query(60, ge=5, le=1440, description="分析最近多少分钟的数据"),
    token: str = Depends(verify_admin_token),
):
    """从数据库历史日志分析各阶段性能瓶颈。

    解析 request_logs.response_body 中的 perf_trace，
    返回分阶段百分位耗时、Per-Token 统计、时间线、慢请求排行。
    """
    return await db.get_performance_history(minutes=minutes, limit=1000)


@router.get("/api/logs")
async def get_logs(
    page: int = 1,
    page_size: int = 10,
    token: str = Depends(verify_admin_token)
):
    """Get paginated lightweight request logs and today's error summary."""
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 10), 100))
    payload = await db.get_logs_paginated(
        page=page,
        page_size=page_size,
        include_payload=False,
    )
    logs = payload.get("items") or []
    error_summary = await db.get_today_error_summary()

    return {
        "success": True,
        "items": [{
            "id": log.get("id"),
            "token_id": log.get("token_id"),
            "token_email": log.get("token_email"),
            "token_username": log.get("token_username"),
            "operation": log.get("operation"),
            "proxy_source": log.get("proxy_source"),
            "status_code": log.get("status_code"),
            "duration": log.get("duration"),
            "status_text": log.get("status_text") or "",
            "progress": log.get("progress") or 0,
            "created_at": log.get("created_at"),
            "updated_at": log.get("updated_at")
        } for log in logs],
        "pagination": {
            "page": payload.get("page", page),
            "page_size": payload.get("page_size", page_size),
            "total": payload.get("total", 0),
            "total_pages": payload.get("total_pages", 1),
        },
        "today_error_summary": error_summary,
    }


@router.get("/api/logs/{log_id}")
async def get_log_detail(
    log_id: int,
    token: str = Depends(verify_admin_token)
):
    """Get single request log detail (payload loaded on demand)"""
    log = await db.get_log_detail(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="日志不存在")

    return {
        "id": log.get("id"),
        "token_id": log.get("token_id"),
        "token_email": log.get("token_email"),
        "token_username": log.get("token_username"),
        "operation": log.get("operation"),
        "status_code": log.get("status_code"),
        "duration": log.get("duration"),
        "status_text": log.get("status_text") or "",
        "progress": log.get("progress") or 0,
        "created_at": log.get("created_at"),
        "updated_at": log.get("updated_at"),
        "request_body": log.get("request_body"),
        "response_body": log.get("response_body")
    }


@router.get("/api/docs/readme")
async def get_readme_document(token: str = Depends(verify_admin_token)):
    """Get README markdown for management panel preview."""
    _ = token
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "docs" / "README.md",
        project_root / "README.md",
    ]

    for file_path in candidates:
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            markdown = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            markdown = file_path.read_text(encoding="utf-8-sig")

        try:
            rel_path = str(file_path.relative_to(project_root)).replace("\\", "/")
        except Exception:
            rel_path = file_path.name

        return {
            "success": True,
            "path": rel_path,
            "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "markdown": markdown,
        }

    raise HTTPException(status_code=404, detail="README.md not found")


@router.get("/api/version")
async def get_project_version(token: str = Depends(verify_admin_token)):
    """Get project version info from config/version.json."""
    project_root = Path(__file__).resolve().parents[2]
    version_path = project_root / "config" / "version.json"

    if not version_path.exists():
        raise HTTPException(status_code=404, detail="version.json not found")

    try:
        payload = json.loads(version_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"version.json 解析失败: {str(e)}")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="version.json 格式无效")

    return {
        "success": True,
        "version": {
            "version": str(payload.get("version") or "").strip(),
            "build": str(payload.get("build") or "").strip(),
            "release_date": str(payload.get("release_date") or "").strip(),
            "channel": str(payload.get("channel") or "").strip(),
            "notes": str(payload.get("notes") or "").strip(),
            "path": str(version_path.relative_to(project_root)).replace("\\", "/"),
        },
    }


@router.get("/api/logs/storage")
async def get_log_storage(token: str = Depends(verify_admin_token)):
    """Get request log storage stats."""
    return await db.get_log_storage_stats()


@router.post("/api/logs/cleanup")
async def cleanup_logs(
    request: LogCleanupRequest,
    token: str = Depends(verify_admin_token),
):
    """Cleanup request logs by retention policy and reclaim space."""
    try:
        result = await db.cleanup_request_logs(
            older_than_days=request.older_than_days,
            keep_latest=request.keep_latest,
            trim_payloads=request.trim_payloads,
            vacuum=request.vacuum,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/logs")
async def clear_logs(token: str = Depends(verify_admin_token)):
    """Clear all logs"""
    try:
        result = await db.clear_all_logs()
        return {"success": True, "message": "所有日志已清空", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/admin/config")
async def get_admin_config(token: str = Depends(verify_admin_token)):
    """Get admin configuration"""
    admin_config = await db.get_admin_config()

    return {
        "admin_username": admin_config.username,
        "api_key": admin_config.api_key,
        "error_ban_threshold": admin_config.error_ban_threshold,
        "debug_enabled": config.debug_enabled  # Return actual debug status
    }


@router.post("/api/admin/config")
async def update_admin_config(
    request: UpdateAdminConfigRequest,
    token: str = Depends(verify_admin_token)
):
    """Update admin configuration (error_ban_threshold)"""
    # Update error_ban_threshold in database
    await db.update_admin_config(error_ban_threshold=request.error_ban_threshold)

    return {"success": True, "message": "配置更新成功"}


@router.post("/api/admin/password")
async def update_admin_password(
    request: ChangePasswordRequest,
    token: str = Depends(verify_admin_token)
):
    """Update admin password"""
    return await change_password(request, token)


@router.post("/api/admin/apikey")
async def update_api_key(
    request: UpdateAPIKeyRequest,
    token: str = Depends(verify_admin_token)
):
    """Update API key (for external API calls, NOT for admin login)"""
    # Update API key in database
    await db.update_admin_config(api_key=request.new_api_key)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()

    return {"success": True, "message": "API Key更新成功"}


@router.post("/api/admin/debug")
async def update_debug_config(
    request: UpdateDebugConfigRequest,
    token: str = Depends(verify_admin_token)
):
    """Update debug configuration"""
    try:
        # Persist to database so the choice survives service restart.
        await db.update_debug_config(enabled=request.enabled)

        # Hot reload: sync database value to in-memory config.
        await db.reload_config_to_memory()

        status = "enabled" if request.enabled else "disabled"
        return {
            "success": True,
            "message": f"Debug mode {status}",
            "enabled": config.debug_enabled,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update debug config: {str(e)}")


@router.get("/api/generation/timeout")
async def get_generation_timeout(token: str = Depends(verify_admin_token)):
    """Get generation timeout configuration"""
    return await get_generation_config(token)


@router.post("/api/generation/timeout")
async def update_generation_timeout(
    request: GenerationConfigRequest,
    token: str = Depends(verify_admin_token)
):
    """Update generation timeout configuration"""
    await db.update_generation_config(request.image_timeout, request.video_timeout)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()

    return {"success": True, "message": "生成配置更新成功"}


# ========== AT Auto Refresh Config ==========

@router.get("/api/token-refresh/config")
async def get_token_refresh_config(token: str = Depends(verify_admin_token)):
    """Get token refresh configuration switches."""
    return {
        "success": True,
        "config": {
            "at_auto_refresh_enabled": True,  # Flow2API默认启用AT自动刷新
            "reauth_cookie_invalid_auto_login_enabled": bool(
                config.reauth_cookie_invalid_auto_login_enabled
            ),
        }
    }


@router.post("/api/token-refresh/enabled")
async def update_token_refresh_enabled(
    request: dict,
    token: str = Depends(verify_admin_token)
):
    """Update token refresh related switches."""
    # AT 自动刷新为固定开启（仅保留前端开关兼容，不支持关闭）。
    requested_at_enabled = request.get("at_auto_refresh_enabled")
    if requested_at_enabled is None and "enabled" in request:
        requested_at_enabled = request.get("enabled")

    # Cookie失效触发账号池自动登录开关
    if "reauth_cookie_invalid_auto_login_enabled" in request:
        requested_reauth_auto_login_enabled = bool(
            request.get("reauth_cookie_invalid_auto_login_enabled")
        )
        try:
            config.update_flow_switches(
                reauth_cookie_invalid_auto_login_enabled=requested_reauth_auto_login_enabled
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"保存 Cookie失效自动登录配置失败: {str(e)}",
            )

    at_msg = "AT自动刷新固定启用"
    if requested_at_enabled is False:
        at_msg = "AT自动刷新固定启用，忽略关闭请求"

    return {
        "success": True,
        "message": at_msg,
        "config": {
            "at_auto_refresh_enabled": True,
            "reauth_cookie_invalid_auto_login_enabled": bool(
                config.reauth_cookie_invalid_auto_login_enabled
            ),
        },
    }


def _sync_runtime_cache_config():
    from . import routes
    if routes.generation_handler and routes.generation_handler.file_cache:
        routes.generation_handler.file_cache.set_timeout(config.cache_timeout)

# ========== Cache Configuration Endpoints ==========

@router.get("/api/cache/config")
async def get_cache_config(token: str = Depends(verify_admin_token)):
    """Get cache configuration"""
    cache_config = await db.get_cache_config()

    # Calculate effective base URL
    effective_base_url = cache_config.cache_base_url if cache_config.cache_base_url else f"http://127.0.0.1:8000"

    return {
        "success": True,
        "config": {
            "enabled": cache_config.cache_enabled,
            "timeout": cache_config.cache_timeout,
            "base_url": cache_config.cache_base_url or "",
            "effective_base_url": effective_base_url
        }
    }


@router.post("/api/cache/enabled")
async def update_cache_enabled(
    request: dict,
    token: str = Depends(verify_admin_token)
):
    """Update cache enabled status"""
    enabled = request.get("enabled", False)
    await db.update_cache_config(enabled=enabled)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()
    _sync_runtime_cache_config()

    return {"success": True, "message": f"缓存已{'启用' if enabled else '禁用'}"}


@router.post("/api/cache/config")
async def update_cache_config_full(
    request: dict,
    token: str = Depends(verify_admin_token)
):
    """Update complete cache configuration"""
    enabled = request.get("enabled")
    timeout = request.get("timeout")
    base_url = request.get("base_url")

    if timeout is not None:
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="缓存超时时间必须为整数")
        if timeout < 0:
            raise HTTPException(status_code=400, detail="缓存超时时间不能小于 0")

    await db.update_cache_config(enabled=enabled, timeout=timeout, base_url=base_url)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()
    _sync_runtime_cache_config()

    return {"success": True, "message": "缓存配置更新成功"}


@router.post("/api/cache/base-url")
async def update_cache_base_url(
    request: dict,
    token: str = Depends(verify_admin_token)
):
    """Update cache base URL"""
    base_url = request.get("base_url", "")
    await db.update_cache_config(base_url=base_url)

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()
    _sync_runtime_cache_config()

    return {"success": True, "message": "缓存Base URL更新成功"}


@router.post("/api/captcha/config")
async def update_captcha_config(
    request: dict,
    token: str = Depends(verify_admin_token)
):
    """Update captcha configuration"""
    from ..services.browser_captcha import validate_browser_proxy_url

    captcha_method = request.get("captcha_method")
    yescaptcha_api_key = request.get("yescaptcha_api_key")
    yescaptcha_base_url = request.get("yescaptcha_base_url")
    capmonster_api_key = request.get("capmonster_api_key")
    capmonster_base_url = request.get("capmonster_base_url")
    ezcaptcha_api_key = request.get("ezcaptcha_api_key")
    ezcaptcha_base_url = request.get("ezcaptcha_base_url")
    capsolver_api_key = request.get("capsolver_api_key")
    capsolver_base_url = request.get("capsolver_base_url")
    captcha_priority_order = normalize_captcha_priority_order(
        request.get("captcha_priority_order", request.get("captcha_method"))
    )
    captcha_method = captcha_priority_order[0]
    remote_browser_base_url = request.get("remote_browser_base_url")
    remote_browser_api_key = request.get("remote_browser_api_key")
    remote_browser_timeout = request.get("remote_browser_timeout", 60)
    remote_browser_proxy_enabled = request.get("remote_browser_proxy_enabled", False)
    browser_proxy_enabled = request.get("browser_proxy_enabled", False)
    browser_proxy_url = request.get("browser_proxy_url", "")
    browser_count = request.get("browser_count", 1)

    # 验证浏览器代理URL格式
    if browser_proxy_enabled and browser_proxy_url:
        is_valid, error_msg = validate_browser_proxy_url(browser_proxy_url)
        if not is_valid:
            return {"success": False, "message": error_msg}

    if remote_browser_base_url:
        try:
            remote_browser_base_url = _normalize_http_base_url(remote_browser_base_url)
        except RuntimeError as e:
            return {"success": False, "message": str(e)}

    try:
        remote_browser_timeout = max(5, int(remote_browser_timeout or 60))
    except Exception:
        return {"success": False, "message": "远程打码超时时间必须是整数秒"}

    await db.update_captcha_config(
        captcha_method=captcha_method,
        yescaptcha_api_key=yescaptcha_api_key,
        yescaptcha_base_url=yescaptcha_base_url,
        capmonster_api_key=capmonster_api_key,
        capmonster_base_url=capmonster_base_url,
        ezcaptcha_api_key=ezcaptcha_api_key,
        ezcaptcha_base_url=ezcaptcha_base_url,
        capsolver_api_key=capsolver_api_key,
        capsolver_base_url=capsolver_base_url,
        captcha_priority_order=captcha_priority_order,
        remote_browser_base_url=remote_browser_base_url,
        remote_browser_api_key=remote_browser_api_key,
        remote_browser_timeout=remote_browser_timeout,
        remote_browser_proxy_enabled=remote_browser_proxy_enabled,
        browser_proxy_enabled=browser_proxy_enabled,
        browser_proxy_url=browser_proxy_url if browser_proxy_enabled else None,
        browser_count=max(1, int(browser_count)) if browser_count else 1
    )

    # 配置包含 browser 时，热重载浏览器数量配置
    if "browser" in captcha_priority_order:
        try:
            from ..services.browser_captcha import BrowserCaptchaService
            service = await BrowserCaptchaService.get_instance(db)
            await service.reload_browser_count()
        except Exception:
            pass

    # 🔥 Hot reload: sync database config to memory
    await db.reload_config_to_memory()

    return {"success": True, "message": "验证码配置更新成功"}


@router.get("/api/captcha/config")
async def get_captcha_config(token: str = Depends(verify_admin_token)):
    """Get captcha configuration"""
    captcha_config = await db.get_captcha_config()
    return {
        "captcha_method": captcha_config.captcha_method,
        "captcha_priority_order": captcha_config.captcha_priority_order,
        "yescaptcha_api_key": captcha_config.yescaptcha_api_key,
        "yescaptcha_base_url": captcha_config.yescaptcha_base_url,
        "capmonster_api_key": captcha_config.capmonster_api_key,
        "capmonster_base_url": captcha_config.capmonster_base_url,
        "ezcaptcha_api_key": captcha_config.ezcaptcha_api_key,
        "ezcaptcha_base_url": captcha_config.ezcaptcha_base_url,
        "capsolver_api_key": captcha_config.capsolver_api_key,
        "capsolver_base_url": captcha_config.capsolver_base_url,
        "remote_browser_base_url": captcha_config.remote_browser_base_url,
        "remote_browser_api_key": captcha_config.remote_browser_api_key,
        "remote_browser_timeout": captcha_config.remote_browser_timeout,
        "remote_browser_proxy_enabled": captcha_config.remote_browser_proxy_enabled,
        "browser_proxy_enabled": captcha_config.browser_proxy_enabled,
        "browser_proxy_url": captcha_config.browser_proxy_url or "",
        "browser_count": captcha_config.browser_count
    }


@router.post("/api/captcha/score-test")
async def test_captcha_score(
    request: Optional[CaptchaScoreTestRequest] = None,
    token: str = Depends(verify_admin_token)
):
    """使用当前打码方式获取 token，并提交到 antcpt 校验分数。"""
    req = request or CaptchaScoreTestRequest()
    website_url = (req.website_url or "https://antcpt.com/score_detector/").strip()
    website_key = (req.website_key or "6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf").strip()
    action = (req.action or "homepage").strip()
    verify_url = (req.verify_url or "https://antcpt.com/score_detector/verify.php").strip()
    enterprise = bool(req.enterprise)

    started_at = time.time()
    captcha_config = await db.get_captcha_config()
    captcha_method = (captcha_config.captcha_method or config.captcha_method or "").strip().lower()
    remote_browser_proxy_enabled = bool(getattr(captcha_config, "remote_browser_proxy_enabled", False))
    browser_proxy_enabled = bool(captcha_config.browser_proxy_enabled)
    browser_proxy_url = captcha_config.browser_proxy_url or ""

    token_value: Optional[str] = None
    fingerprint: Optional[Dict[str, Any]] = None
    token_elapsed_ms = 0
    verify_elapsed_ms = 0
    verify_http_status = None
    verify_result: Dict[str, Any] = {}
    verify_headers: Dict[str, str] = {}
    verify_proxy_used = False
    verify_proxy_source = "none"
    verify_proxy_url = ""
    verify_impersonate = "chrome120"
    page_verify_only = captcha_method in {"browser", "personal", "remote_browser"}
    verify_mode = "browser_page" if page_verify_only else "server_post"

    try:
        token_start = time.time()
        if captcha_method == "browser":
            from ..services.browser_captcha import BrowserCaptchaService
            service = await BrowserCaptchaService.get_instance(db)
            score_payload, browser_id = await service.get_custom_score(
                website_url=website_url,
                website_key=website_key,
                verify_url=verify_url,
                action=action,
                enterprise=enterprise
            )
            if isinstance(score_payload, dict):
                token_value = score_payload.get("token")
                verify_elapsed_ms = int(score_payload.get("verify_elapsed_ms") or 0)
                verify_http_status = score_payload.get("verify_http_status")
                verify_result = score_payload.get("verify_result") if isinstance(score_payload.get("verify_result"), dict) else {}
                verify_mode = score_payload.get("verify_mode") or "browser_page"
                score_token_elapsed = score_payload.get("token_elapsed_ms")
                if isinstance(score_token_elapsed, (int, float)):
                    token_elapsed_ms = int(score_token_elapsed)
            if token_value:
                fingerprint = await service.get_fingerprint(browser_id)
                verify_proxy_used = bool(browser_proxy_enabled and browser_proxy_url)
                verify_proxy_source = "captcha_browser_proxy" if verify_proxy_used else "browser_direct"
                verify_proxy_url = browser_proxy_url if verify_proxy_used else ""
        elif captcha_method == "personal":
            from ..services.browser_captcha_personal import BrowserCaptchaService
            service = await BrowserCaptchaService.get_instance(db)
            score_payload = await service.get_custom_score(
                website_url=website_url,
                website_key=website_key,
                verify_url=verify_url,
                action=action,
                enterprise=enterprise
            )
            if isinstance(score_payload, dict):
                token_value = score_payload.get("token")
                verify_elapsed_ms = int(score_payload.get("verify_elapsed_ms") or 0)
                verify_http_status = score_payload.get("verify_http_status")
                verify_result = score_payload.get("verify_result") if isinstance(score_payload.get("verify_result"), dict) else {}
                verify_mode = score_payload.get("verify_mode") or "browser_page"
                score_token_elapsed = score_payload.get("token_elapsed_ms")
                if isinstance(score_token_elapsed, (int, float)):
                    token_elapsed_ms = int(score_token_elapsed)
            if token_value:
                fingerprint = service.get_last_fingerprint()
                verify_proxy_used = bool(browser_proxy_enabled and browser_proxy_url)
                verify_proxy_source = "captcha_browser_proxy" if verify_proxy_used else "browser_direct"
                verify_proxy_url = browser_proxy_url if verify_proxy_used else ""
        elif captcha_method == "remote_browser":
            score_payload = await _score_test_with_remote_browser_service(
                website_url=website_url,
                website_key=website_key,
                verify_url=verify_url,
                action=action,
                enterprise=enterprise,
            )
            if isinstance(score_payload, dict):
                if score_payload.get("success") is False:
                    raise RuntimeError(score_payload.get("message") or "远程打码分数测试失败")
                token_value = score_payload.get("token")
                verify_elapsed_ms = int(score_payload.get("verify_elapsed_ms") or 0)
                verify_http_status = score_payload.get("verify_http_status")
                verify_result = score_payload.get("verify_result") if isinstance(score_payload.get("verify_result"), dict) else {}
                verify_mode = score_payload.get("verify_mode") or "remote_browser_page"
                score_token_elapsed = score_payload.get("token_elapsed_ms")
                if isinstance(score_token_elapsed, (int, float)):
                    token_elapsed_ms = int(score_token_elapsed)
                fingerprint = score_payload.get("fingerprint") if isinstance(score_payload.get("fingerprint"), dict) else None
                verify_proxy_source = str(score_payload.get("proxy_source") or "direct")
                verify_proxy_url = str(score_payload.get("proxy_url") or "").strip()
                verify_proxy_used = bool(
                    verify_proxy_url or verify_proxy_source not in {"", "direct", "none"}
                )
        elif captcha_method in SUPPORTED_API_CAPTCHA_METHODS:
            token_value = await _solve_recaptcha_with_api_service(
                method=captcha_method,
                website_url=website_url,
                website_key=website_key,
                action=action,
                enterprise=enterprise
            )
        else:
            return {
                "success": False,
                "message": f"当前打码方式不支持分数测试: {captcha_method}",
                "captcha_method": captcha_method,
                "website_url": website_url,
                "website_key": website_key,
                "action": action,
                "verify_url": verify_url,
                "enterprise": enterprise,
                "token_acquired": False,
                "elapsed_ms": int((time.time() - started_at) * 1000)
            }
        if token_elapsed_ms <= 0:
            token_elapsed_ms = int((time.time() - token_start) * 1000)

        # 远程有头打码的 custom-score 可能由页面内直接完成校验，
        # 在部分实现里不会显式回传 token，本地按 verify_result 兜底判定。
        if captcha_method == "remote_browser" and not token_value and isinstance(verify_result, dict):
            if verify_result.get("success") is True:
                token_value = verify_result.get("token") or verify_result.get("gRecaptchaResponse") or "__verified_by_remote__"

        if not token_value:
            return {
                "success": False,
                "message": "未获取到 reCAPTCHA token",
                "captcha_method": captcha_method,
                "website_url": website_url,
                "website_key": website_key,
                "action": action,
                "verify_url": verify_url,
                "enterprise": enterprise,
                "token_acquired": False,
                "token_elapsed_ms": token_elapsed_ms,
                "remote_browser_proxy_enabled": remote_browser_proxy_enabled,
                "browser_proxy_enabled": browser_proxy_enabled,
                "browser_proxy_url": browser_proxy_url if browser_proxy_enabled else "",
                "fingerprint": fingerprint,
                "elapsed_ms": int((time.time() - started_at) * 1000)
            }

        if verify_mode == "server_post" and not page_verify_only:
            verify_start = time.time()
            verify_headers = {
                "accept": "application/json, text/javascript, */*; q=0.01",
                "content-type": "application/json",
                "origin": "https://antcpt.com",
                "referer": website_url,
                "x-requested-with": "XMLHttpRequest",
            }
            if isinstance(fingerprint, dict):
                ua = (fingerprint.get("user_agent") or "").strip()
                lang = (fingerprint.get("accept_language") or "").strip()
                sec_ch_ua = (fingerprint.get("sec_ch_ua") or "").strip()
                sec_ch_ua_mobile = (fingerprint.get("sec_ch_ua_mobile") or "").strip()
                sec_ch_ua_platform = (fingerprint.get("sec_ch_ua_platform") or "").strip()

                if ua:
                    verify_headers["user-agent"] = ua
                if lang:
                    verify_headers["accept-language"] = lang if "," in lang else f"{lang},zh;q=0.9"
                if sec_ch_ua:
                    verify_headers["sec-ch-ua"] = sec_ch_ua
                if sec_ch_ua_mobile:
                    verify_headers["sec-ch-ua-mobile"] = sec_ch_ua_mobile
                if sec_ch_ua_platform:
                    verify_headers["sec-ch-ua-platform"] = sec_ch_ua_platform

            if verify_headers.get("user-agent"):
                for header_name, header_value in _guess_client_hints_from_user_agent(
                    verify_headers.get("user-agent", "")
                ).items():
                    if header_value and not verify_headers.get(header_name):
                        verify_headers[header_name] = header_value
                verify_impersonate = _guess_impersonate_from_user_agent(verify_headers.get("user-agent", ""))

            verify_proxies, verify_proxy_used, verify_proxy_source, verify_proxy_url = (
                await _resolve_score_test_verify_proxy(
                    captcha_method=captcha_method,
                    browser_proxy_enabled=browser_proxy_enabled,
                    browser_proxy_url=browser_proxy_url
                )
            )

            async with AsyncSession() as session:
                verify_resp = await session.post(
                    verify_url,
                    json={"g-recaptcha-response": token_value},
                    headers=verify_headers,
                    proxies=verify_proxies,
                    impersonate=verify_impersonate,
                    timeout=30
                )
            verify_elapsed_ms = int((time.time() - verify_start) * 1000)
            verify_http_status = verify_resp.status_code

            try:
                verify_result = verify_resp.json()
            except Exception:
                verify_result = {"raw": verify_resp.text}
        else:
            verify_headers = {
                "origin": "https://antcpt.com",
                "referer": website_url,
                "x-requested-with": "XMLHttpRequest",
            }
            if isinstance(fingerprint, dict):
                verify_headers.update({
                    "user-agent": fingerprint.get("user_agent", ""),
                    "accept-language": fingerprint.get("accept_language", ""),
                    "sec-ch-ua": fingerprint.get("sec_ch_ua", ""),
                    "sec-ch-ua-mobile": fingerprint.get("sec_ch_ua_mobile", ""),
                    "sec-ch-ua-platform": fingerprint.get("sec_ch_ua_platform", ""),
                })

        verify_success = bool(verify_result.get("success")) if isinstance(verify_result, dict) else False
        score_value = verify_result.get("score") if isinstance(verify_result, dict) else None

        return {
            "success": verify_success,
            "message": "分数校验成功" if verify_success else "分数校验未通过",
            "captcha_method": captcha_method,
            "website_url": website_url,
            "website_key": website_key,
            "action": action,
            "verify_url": verify_url,
            "enterprise": enterprise,
            "token_acquired": True,
            "token_preview": _mask_token(token_value),
            "token_elapsed_ms": token_elapsed_ms,
            "verify_elapsed_ms": verify_elapsed_ms,
            "verify_http_status": verify_http_status,
            "score": score_value,
            "verify_result": verify_result,
            "verify_request_meta": {
                "mode": verify_mode,
                "proxy_used": verify_proxy_used,
                "user_agent": verify_headers.get("user-agent", ""),
                "accept_language": verify_headers.get("accept-language", ""),
                "sec_ch_ua": verify_headers.get("sec-ch-ua", ""),
                "sec_ch_ua_mobile": verify_headers.get("sec-ch-ua-mobile", ""),
                "sec_ch_ua_platform": verify_headers.get("sec-ch-ua-platform", ""),
                "origin": verify_headers.get("origin", ""),
                "referer": verify_headers.get("referer", ""),
                "x_requested_with": verify_headers.get("x-requested-with", ""),
                "proxy_source": verify_proxy_source,
                "proxy_url": verify_proxy_url,
                "impersonate": verify_impersonate,
            },
            "browser_proxy_enabled": browser_proxy_enabled,
            "browser_proxy_url": browser_proxy_url if browser_proxy_enabled else "",
            "fingerprint": fingerprint,
            "elapsed_ms": int((time.time() - started_at) * 1000)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"分数测试失败: {str(e)}",
            "captcha_method": captcha_method,
            "website_url": website_url,
            "website_key": website_key,
            "action": action,
            "verify_url": verify_url,
            "enterprise": enterprise,
            "token_acquired": bool(token_value),
            "token_preview": _mask_token(token_value),
            "token_elapsed_ms": token_elapsed_ms,
            "verify_elapsed_ms": verify_elapsed_ms,
            "verify_http_status": verify_http_status,
            "verify_result": verify_result,
            "verify_request_meta": {
                "mode": verify_mode,
                "proxy_used": verify_proxy_used,
                "user_agent": verify_headers.get("user-agent", ""),
                "accept_language": verify_headers.get("accept-language", ""),
                "sec_ch_ua": verify_headers.get("sec-ch-ua", ""),
                "sec_ch_ua_mobile": verify_headers.get("sec-ch-ua-mobile", ""),
                "sec_ch_ua_platform": verify_headers.get("sec-ch-ua-platform", ""),
                "origin": verify_headers.get("origin", ""),
                "referer": verify_headers.get("referer", ""),
                "x_requested_with": verify_headers.get("x-requested-with", ""),
                "proxy_source": verify_proxy_source,
                "proxy_url": verify_proxy_url,
                "impersonate": verify_impersonate,
            },
            "browser_proxy_enabled": browser_proxy_enabled,
            "browser_proxy_url": browser_proxy_url if browser_proxy_enabled else "",
            "fingerprint": fingerprint,
            "elapsed_ms": int((time.time() - started_at) * 1000)
        }


# ========== Plugin Configuration Endpoints ==========

@router.get("/api/plugin/config")
async def get_plugin_config(request: Request, token: str = Depends(verify_admin_token)):
    """Get plugin configuration"""
    plugin_config = await db.get_plugin_config()

    # Get the actual domain and port from the request
    # This allows the connection URL to reflect the user's actual access path
    host_header = request.headers.get("host", "")

    # Generate connection URL based on actual request
    if host_header:
        # Use the actual domain/IP and port from the request
        connection_url = f"http://{host_header}/api/plugin/update-token"
    else:
        # Fallback to config-based URL
        from ..core.config import config
        server_host = config.server_host
        server_port = config.server_port

        if server_host == "0.0.0.0":
            connection_url = f"http://127.0.0.1:{server_port}/api/plugin/update-token"
        else:
            connection_url = f"http://{server_host}:{server_port}/api/plugin/update-token"

    return {
        "success": True,
        "config": {
            "connection_token": plugin_config.connection_token,
            "connection_url": connection_url,
            "auto_enable_on_update": plugin_config.auto_enable_on_update
        }
    }


@router.post("/api/plugin/config")
async def update_plugin_config(
    request: dict,
    token: str = Depends(verify_admin_token)
):
    """Update plugin configuration"""
    connection_token = request.get("connection_token", "")
    auto_enable_on_update = request.get("auto_enable_on_update", True)  # 默认开启

    # Generate random token if empty
    if not connection_token:
        connection_token = secrets.token_urlsafe(32)

    await db.update_plugin_config(
        connection_token=connection_token,
        auto_enable_on_update=auto_enable_on_update
    )

    return {
        "success": True,
        "message": "插件配置更新成功",
        "connection_token": connection_token,
        "auto_enable_on_update": auto_enable_on_update
    }


@router.post("/api/plugin/update-token")
async def plugin_update_token(request: dict, authorization: Optional[str] = Header(None)):
    """Receive token update from Chrome extension (no admin auth required, uses connection_token)"""
    # Verify connection token
    plugin_config = await db.get_plugin_config()

    # Extract token from Authorization header
    provided_token = None
    if authorization:
        if authorization.startswith("Bearer "):
            provided_token = authorization[7:]
        else:
            provided_token = authorization

    # Check if token matches
    if not plugin_config.connection_token or provided_token != plugin_config.connection_token:
        raise HTTPException(status_code=401, detail="Invalid connection token")

    # Extract session token from request
    session_token = request.get("session_token")

    if not session_token:
        raise HTTPException(status_code=400, detail="Missing session_token")

    # Step 1: Convert ST to AT to get user info (including email)
    try:
        result = await token_manager.flow_client.st_to_at(session_token)
        at = result["access_token"]
        expires = result.get("expires")
        user_info = result.get("user", {})
        email = user_info.get("email", "")

        if not email:
            raise HTTPException(status_code=400, detail="Failed to get email from session token")

        # Parse expiration time
        from datetime import datetime
        at_expires = None
        if expires:
            try:
                at_expires = datetime.fromisoformat(expires.replace('Z', '+00:00'))
            except:
                pass

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session token: {str(e)}")

    # Step 2: Check if token with this email exists
    existing_token = await db.get_token_by_email(email)

    if existing_token:
        # Update existing token
        try:
            # Update token
            await token_manager.update_token(
                token_id=existing_token.id,
                st=session_token,
                at=at,
                at_expires=at_expires
            )

            # Check if auto-enable is enabled and token is disabled
            if plugin_config.auto_enable_on_update and not existing_token.is_active:
                await token_manager.enable_token(existing_token.id)
                return {
                    "success": True,
                    "message": f"Token updated and auto-enabled for {email}",
                    "action": "updated",
                    "auto_enabled": True
                }

            return {
                "success": True,
                "message": f"Token updated for {email}",
                "action": "updated"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update token: {str(e)}")
    else:
        # Add new token
        try:
            new_token = await token_manager.add_token(
                st=session_token,
                remark="Added by Chrome Extension"
            )

            return {
                "success": True,
                "message": f"Token added for {new_token.email}",
                "action": "added",
                "token_id": new_token.id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add token: {str(e)}")
