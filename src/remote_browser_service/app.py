from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Union

import tomli
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from CommonFramePackage.proxy_pool import ProxyPoolRepository, ProxyPoolService

from ..core.database import Database
from ..core.logger import debug_logger
from ..services.browser_captcha import BrowserCaptchaService
from ..services.proxy_manager import ProxyManager

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = REPO_ROOT / "tmp"
TMP_TOKEN_DIR = TMP_DIR / "Token"
DASHBOARD_FILE = Path(__file__).with_name("dashboard.html")
DEFAULT_CONFIG_FILE = REPO_ROOT / "config" / "remote_browser_service.toml"


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    try:
        value = int(raw) if raw else int(default)
    except Exception:
        value = int(default)
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _get_api_key() -> str:
    api_key = str(os.getenv("REMOTE_BROWSER_API_KEY", "") or "").strip()
    if api_key:
        return api_key

    config_path = Path(
        str(os.getenv("REMOTE_BROWSER_CONFIG_PATH", "") or "").strip()
    ) if str(os.getenv("REMOTE_BROWSER_CONFIG_PATH", "") or "").strip() else DEFAULT_CONFIG_FILE

    if config_path.exists() and config_path.is_file():
        try:
            with config_path.open("rb") as fh:
                payload = tomli.load(fh) or {}
            api_key = str(
                (payload.get("remote_browser_service", {}) or {}).get("api_key") or ""
            ).strip()
        except Exception as e:
            raise RuntimeError(f"读取远程服务配置文件失败: {config_path} -> {e}") from e

    if not api_key:
        raise RuntimeError(
            f"REMOTE_BROWSER_API_KEY 未配置，且配置文件中未找到 api_key: {config_path}"
        )
    return api_key


def _extract_email_from_token_file(path: Path) -> Optional[str]:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    email = str((payload or {}).get("email") or "").strip()
    if "@" in email:
        return email

    stem = path.stem
    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            stem = prefix
    guessed = stem.replace("_at_", "@").strip()
    if "@" in guessed:
        return guessed
    return None


async def _resolve_credential_email(credential_name: str, mode: str) -> Optional[str]:
    if str(mode or "").strip().lower() != "antigravity":
        return None
    safe_name = Path(str(credential_name or "")).name
    if not safe_name:
        return None
    file_path = TMP_TOKEN_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        return None
    return _extract_email_from_token_file(file_path)


@dataclass
class SessionEntry:
    session_id: str
    browser_ref: Optional[Union[int, str]]
    project_id: str
    action: str
    created_at: float
    token_id: Optional[int] = None
    proxy_source: str = "direct"
    proxy_url: Optional[str] = None


@dataclass
class TaskEntry:
    task_id: str
    task_type: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None
    project_id: Optional[str] = None
    action: Optional[str] = None
    token_id: Optional[int] = None
    session_id: Optional[str] = None
    browser_ref: Optional[str] = None
    proxy_source: str = "direct"
    proxy_url: Optional[str] = None
    error: Optional[str] = None
    note: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "project_id": self.project_id,
            "action": self.action,
            "token_id": self.token_id,
            "session_id": self.session_id,
            "browser_ref": self.browser_ref,
            "proxy_source": self.proxy_source,
            "proxy_url": self.proxy_url,
            "error": self.error,
            "note": self.note,
        }
        payload.update(self.extra or {})
        return payload


class SessionRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, SessionEntry] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        browser_ref: Optional[Union[int, str]],
        project_id: str,
        action: str,
        *,
        token_id: Optional[int] = None,
        proxy_source: str = "direct",
        proxy_url: Optional[str] = None,
    ) -> SessionEntry:
        entry = SessionEntry(
            session_id=uuid.uuid4().hex,
            browser_ref=browser_ref,
            project_id=str(project_id or "").strip(),
            action=str(action or "").strip(),
            created_at=time.time(),
            token_id=token_id,
            proxy_source=str(proxy_source or "direct").strip() or "direct",
            proxy_url=str(proxy_url or "").strip() or None,
        )
        async with self._lock:
            self._items[entry.session_id] = entry
        return entry

    async def pop(self, session_id: str) -> Optional[SessionEntry]:
        key = str(session_id or "").strip()
        if not key:
            return None
        async with self._lock:
            return self._items.pop(key, None)

    async def count(self) -> int:
        async with self._lock:
            return len(self._items)

    async def snapshot(self) -> List[Dict[str, Any]]:
        now = time.time()
        async with self._lock:
            items = list(self._items.values())
        return [
            {
                "session_id": item.session_id,
                "browser_ref": str(item.browser_ref) if item.browser_ref is not None else None,
                "project_id": item.project_id,
                "action": item.action,
                "token_id": item.token_id,
                "proxy_source": item.proxy_source,
                "proxy_url": item.proxy_url,
                "age_seconds": round(max(0.0, now - float(item.created_at)), 1),
                "created_at": datetime.fromtimestamp(item.created_at, tz=timezone.utc).isoformat(),
            }
            for item in sorted(items, key=lambda x: x.created_at, reverse=True)
        ]

    async def drop_stale(self, ttl_seconds: int) -> List[SessionEntry]:
        now = time.time()
        removed: List[SessionEntry] = []
        async with self._lock:
            stale_keys = [
                key
                for key, entry in self._items.items()
                if now - float(entry.created_at) >= float(ttl_seconds)
            ]
            for key in stale_keys:
                entry = self._items.pop(key, None)
                if entry is not None:
                    removed.append(entry)
        return removed


class SolveRequest(BaseModel):
    project_id: str = Field(..., min_length=1, description="Flow 项目 ID")
    action: str = Field(default="IMAGE_GENERATION", description="reCAPTCHA action")
    token_id: Optional[int] = Field(default=None, description="业务 token_id，用于从代理池或 token 配置读取代理")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_id": "beac061a-cf4c-483f-9d77-4b2d55c29bae",
                "action": "IMAGE_GENERATION",
                "token_id": 123,
            }
        }
    }


class FinishRequest(BaseModel):
    status: str = Field(default="success", description="上游请求结束状态")


class ErrorRequest(BaseModel):
    error_reason: str = Field(default="upstream_error", description="错误原因")


class CustomScoreRequest(BaseModel):
    website_url: str = Field(..., description="测试页面 URL")
    website_key: str = Field(..., description="reCAPTCHA site key")
    verify_url: str = Field(..., description="页面内分数校验接口 URL")
    action: str = Field(default="homepage", description="reCAPTCHA action")
    enterprise: bool = Field(default=False, description="是否使用 enterprise 模式")

    model_config = {
        "json_schema_extra": {
            "example": {
                "website_url": "https://antcpt.com/score_detector/",
                "website_key": "6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf",
                "verify_url": "https://antcpt.com/score_detector/verify.php",
                "action": "homepage",
                "enterprise": False,
            }
        }
    }


class RemoteBrowserRuntime:
    def __init__(self) -> None:
        self.session_registry = SessionRegistry()
        self.service: Optional[BrowserCaptchaService] = None
        self.db: Optional[Database] = None
        self.proxy_manager: Optional[ProxyManager] = None
        self.proxy_pool_service: Optional[ProxyPoolService] = None
        self._reaper_task: Optional[asyncio.Task] = None
        self._task_lock = asyncio.Lock()
        self._active_tasks: Dict[str, TaskEntry] = {}
        self._recent_tasks: Deque[TaskEntry] = deque(
            maxlen=_env_int("REMOTE_BROWSER_TASK_HISTORY_LIMIT", default=200, minimum=20, maximum=2000)
        )
        self.session_ttl_seconds = _env_int(
            "REMOTE_BROWSER_SESSION_TTL_SECONDS",
            default=3600,
            minimum=60,
            maximum=24 * 3600,
        )
        self.cleanup_interval_seconds = _env_int(
            "REMOTE_BROWSER_SESSION_REAPER_INTERVAL_SECONDS",
            default=60,
            minimum=15,
            maximum=3600,
        )

    async def startup(self) -> None:
        _ = _get_api_key()

        db = Database()
        await db.init_db()
        proxy_manager = ProxyManager(db)
        proxy_pool_repo = ProxyPoolRepository(credential_email_resolver=_resolve_credential_email)
        proxy_pool_service = ProxyPoolService(proxy_pool_repo)
        await proxy_pool_service.initialize()
        proxy_manager.set_proxy_pool_service(proxy_pool_service)

        service = await BrowserCaptchaService.get_instance(db=db)
        service.db = db
        service.set_external_proxy_resolver(self.resolve_proxy_for_token)

        base_user_data_dir = str(os.getenv("REMOTE_BROWSER_USER_DATA_DIR", "") or "").strip()
        if base_user_data_dir:
            service.base_user_data_dir = base_user_data_dir

        browser_count_raw = str(os.getenv("REMOTE_BROWSER_BROWSER_COUNT", "") or "").strip()
        if browser_count_raw:
            browser_count = _env_int(
                "REMOTE_BROWSER_BROWSER_COUNT",
                default=getattr(service, "_browser_count", 1) or 1,
                minimum=1,
                maximum=20,
            )
            service._browser_count = browser_count
            service._token_semaphore = asyncio.Semaphore(browser_count)
        else:
            await service._load_browser_count()
            browser_count = int(getattr(service, "_browser_count", 1) or 1)

        await service._ensure_idle_reaper()

        self.db = db
        self.proxy_manager = proxy_manager
        self.proxy_pool_service = proxy_pool_service
        self.service = service
        self._reaper_task = asyncio.create_task(self._session_reaper_loop())

        if _env_bool("REMOTE_BROWSER_WARMUP_ON_STARTUP", default=False):
            try:
                await service.warmup_browser_slots()
            except Exception as e:
                debug_logger.log_warning(f"[RemoteBrowser] 预热浏览器失败: {e}")

        debug_logger.log_info(
            f"[RemoteBrowser] 服务已启动: browser_count={browser_count}, session_ttl={self.session_ttl_seconds}s, proxy_pool={'yes' if self.proxy_pool_service else 'no'}"
        )

    async def shutdown(self) -> None:
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except Exception:
                pass
            self._reaper_task = None

        if self.service:
            try:
                await self.service.close()
            except Exception as e:
                debug_logger.log_warning(f"[RemoteBrowser] 服务关闭时清理浏览器失败: {e}")

        debug_logger.log_info("[RemoteBrowser] 服务已关闭")

    async def _session_reaper_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                removed = await self.session_registry.drop_stale(self.session_ttl_seconds)
                for entry in removed:
                    await self._append_completed_task(
                        task_type="session_reaper",
                        status="expired",
                        project_id=entry.project_id,
                        action=entry.action,
                        token_id=entry.token_id,
                        session_id=entry.session_id,
                        browser_ref=str(entry.browser_ref) if entry.browser_ref is not None else None,
                        proxy_source=entry.proxy_source,
                        proxy_url=entry.proxy_url,
                        note=f"会话超过 {self.session_ttl_seconds}s 被清理",
                    )
                if removed:
                    debug_logger.log_info(f"[RemoteBrowser] 已清理过期会话: removed={len(removed)}")
            except asyncio.CancelledError:
                return
            except Exception as e:
                debug_logger.log_warning(f"[RemoteBrowser] 会话清理任务异常: {e}")

    async def resolve_proxy_for_token(self, token_id: Optional[int]) -> tuple[Optional[str], str]:
        manager = self.proxy_manager
        if manager is None or self.db is None:
            return None, "direct"
        try:
            captcha_config = await self.db.get_captcha_config()
            if not bool(getattr(captcha_config, "remote_browser_proxy_enabled", False)):
                return None, "direct"
        except Exception as e:
            debug_logger.log_warning(f"[RemoteBrowser] 读取远程代理开关失败，按直连处理: {e}")
            return None, "direct"
        return await manager.select_proxy_with_source_for_token_id(token_id=token_id)

    async def _start_task(
        self,
        *,
        task_type: str,
        project_id: Optional[str] = None,
        action: Optional[str] = None,
        token_id: Optional[int] = None,
        session_id: Optional[str] = None,
        browser_ref: Optional[str] = None,
        proxy_source: str = "direct",
        proxy_url: Optional[str] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> TaskEntry:
        entry = TaskEntry(
            task_id=uuid.uuid4().hex,
            task_type=task_type,
            status="running",
            started_at=_utc_iso_now(),
            project_id=project_id,
            action=action,
            token_id=token_id,
            session_id=session_id,
            browser_ref=browser_ref,
            proxy_source=proxy_source,
            proxy_url=proxy_url,
            note=note,
            extra=dict(extra or {}),
        )
        async with self._task_lock:
            self._active_tasks[entry.task_id] = entry
        return entry

    async def _finish_task(
        self,
        task_id: str,
        *,
        status: str,
        error: Optional[str] = None,
        note: Optional[str] = None,
        session_id: Optional[str] = None,
        browser_ref: Optional[str] = None,
        proxy_source: Optional[str] = None,
        proxy_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with self._task_lock:
            entry = self._active_tasks.pop(task_id, None)
            if entry is None:
                return
            entry.status = status
            entry.finished_at = _utc_iso_now()
            started_dt = datetime.fromisoformat(entry.started_at)
            finished_dt = datetime.fromisoformat(entry.finished_at)
            entry.duration_ms = max(0, int((finished_dt - started_dt).total_seconds() * 1000))
            if error:
                entry.error = error
            if note:
                entry.note = note
            if session_id:
                entry.session_id = session_id
            if browser_ref:
                entry.browser_ref = browser_ref
            if proxy_source:
                entry.proxy_source = proxy_source
            if proxy_url is not None:
                entry.proxy_url = proxy_url
            if extra:
                entry.extra.update(extra)
            self._recent_tasks.appendleft(entry)

    async def _append_completed_task(self, **kwargs: Any) -> None:
        entry = TaskEntry(
            task_id=uuid.uuid4().hex,
            task_type=str(kwargs.pop("task_type", "event") or "event"),
            status=str(kwargs.pop("status", "success") or "success"),
            started_at=_utc_iso_now(),
            finished_at=_utc_iso_now(),
            **kwargs,
        )
        started_dt = datetime.fromisoformat(entry.started_at)
        finished_dt = datetime.fromisoformat(entry.finished_at)
        entry.duration_ms = max(0, int((finished_dt - started_dt).total_seconds() * 1000))
        async with self._task_lock:
            self._recent_tasks.appendleft(entry)

    async def snapshot_overview(self) -> Dict[str, Any]:
        service = self.service
        stats = service.get_stats() if service else {}
        proxies = await self.snapshot_proxies(limit=50)
        active_sessions = await self.session_registry.snapshot()
        async with self._task_lock:
            active_tasks = [entry.to_dict() for entry in self._active_tasks.values()]
            recent_tasks = [entry.to_dict() for entry in list(self._recent_tasks)]

        proxy_items = proxies.get("items") or []
        available_proxy_count = len([item for item in proxy_items if not item.get("disabled")])
        tested_ok_proxy_count = len(
            [item for item in proxy_items if not item.get("disabled") and item.get("last_test_ok") in (1, True)]
        )

        return {
            "success": True,
            "service": "remote_browser",
            "generated_at": _utc_iso_now(),
            "session_ttl_seconds": self.session_ttl_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "stats": stats,
            "active_sessions": active_sessions,
            "active_session_count": len(active_sessions),
            "active_tasks": active_tasks,
            "active_task_count": len(active_tasks),
            "recent_tasks": recent_tasks,
            "recent_task_count": len(recent_tasks),
            "proxy_pool": {
                "total": proxies.get("total", 0),
                "items": proxy_items,
                "available_count": available_proxy_count,
                "tested_ok_count": tested_ok_proxy_count,
            },
        }

    async def snapshot_proxies(self, limit: int = 50) -> Dict[str, Any]:
        service = self.proxy_pool_service
        if service is None:
            return {"total": 0, "items": [], "offset": 0, "limit": limit}
        try:
            return await service.list_proxies(offset=0, limit=limit, search=None, host=None)
        except Exception as e:
            debug_logger.log_warning(f"[RemoteBrowser] 加载代理池失败: {e}")
            return {"total": 0, "items": [], "offset": 0, "limit": limit, "error": str(e)}


runtime = RemoteBrowserRuntime()
bearer_scheme = HTTPBearer(
    scheme_name="RemoteBrowserBearer",
    description="在这里输入 REMOTE_BROWSER_API_KEY，格式为 Bearer <token>",
    auto_error=False,
)


async def verify_remote_browser_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    expected = _get_api_key()
    if credentials is None or str(credentials.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing authorization")
    token = str(credentials.credentials or "").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token


def _require_runtime_service() -> BrowserCaptchaService:
    if runtime.service is None:
        raise HTTPException(status_code=503, detail="Remote browser service is not ready")
    return runtime.service


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.startup()
    try:
        yield
    finally:
        await runtime.shutdown()


app = FastAPI(
    title="Flow2API Remote Browser Service",
    version="0.2.0",
    summary="独立部署的远程有头浏览器打码服务",
    description=(
        "用于给 Flow2API 提供 remote_browser 打码能力。"
        "支持获取 Flow reCAPTCHA token、页面内分数测试、代理池接入、会话回调与运行状态面板。"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "dashboard", "description": "远程打码服务状态面板与运行态接口"},
        {"name": "system", "description": "健康检查与服务状态"},
        {"name": "captcha", "description": "远程打码与页面分数测试接口"},
        {"name": "session", "description": "远程浏览器会话完成与错误回调接口"},
    ],
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(str(DASHBOARD_FILE))


@app.get("/swagger", include_in_schema=False)
async def swagger_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz", tags=["system"], summary="健康检查")
async def healthz() -> Dict[str, Any]:
    service = runtime.service
    stats = service.get_stats() if service else {}
    return {
        "success": True,
        "service": "remote_browser",
        "configured_browser_count": int(stats.get("configured_browser_count") or 0),
        "busy_browser_count": int(stats.get("busy_browser_count") or 0),
        "active_sessions": await runtime.session_registry.count(),
        "active_tasks": len(runtime._active_tasks),
    }


@app.get(
    "/ui/api/overview",
    tags=["dashboard"],
    summary="获取仪表盘总览",
    description="返回远程打码服务运行状态、浏览器并发、最近任务和代理池概览。",
)
async def dashboard_overview() -> Dict[str, Any]:
    return await runtime.snapshot_overview()


@app.post(
    "/api/v1/solve",
    tags=["captcha"],
    summary="获取 Flow reCAPTCHA token",
    description="使用远程有头浏览器打开 Flow 页面并返回 token、session_id 和浏览器指纹。",
)
async def solve_recaptcha(
    request: SolveRequest,
    _: str = Depends(verify_remote_browser_api_key),
) -> Dict[str, Any]:
    service = _require_runtime_service()
    selected_proxy_url, selected_proxy_source = await runtime.resolve_proxy_for_token(request.token_id)
    task = await runtime._start_task(
        task_type="solve",
        project_id=request.project_id,
        action=request.action,
        token_id=request.token_id,
        proxy_source=selected_proxy_source,
        proxy_url=selected_proxy_url,
    )
    try:
        token, browser_ref = await service.get_token(
            project_id=request.project_id,
            action=request.action,
            token_id=request.token_id,
            proxy_url_override=selected_proxy_url,
        )
        if not token:
            raise RuntimeError("未获取到 reCAPTCHA token")

        fingerprint = await service.get_fingerprint(browser_ref) or {}
        actual_proxy_url = str(fingerprint.get("proxy_url") or selected_proxy_url or "").strip() or None
        actual_proxy_source = selected_proxy_source
        if actual_proxy_url and actual_proxy_source == "direct":
            actual_proxy_source = "browser_config"

        session_entry = await runtime.session_registry.create(
            browser_ref=browser_ref,
            project_id=request.project_id,
            action=request.action,
            token_id=request.token_id,
            proxy_source=actual_proxy_source,
            proxy_url=actual_proxy_url,
        )

        await runtime._finish_task(
            task.task_id,
            status="success",
            session_id=session_entry.session_id,
            browser_ref=str(browser_ref),
            proxy_source=actual_proxy_source,
            proxy_url=actual_proxy_url,
            extra={"fingerprint": fingerprint or {}},
        )
        return {
            "success": True,
            "token": token,
            "session_id": session_entry.session_id,
            "fingerprint": fingerprint or {},
        }
    except HTTPException:
        await runtime._finish_task(task.task_id, status="failed", error="HTTPException")
        raise
    except Exception as e:
        await runtime._finish_task(
            task.task_id,
            status="failed",
            error=str(e),
            proxy_source=selected_proxy_source,
            proxy_url=selected_proxy_url,
        )
        debug_logger.log_error(f"[RemoteBrowser] solve 失败: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.post(
    "/api/v1/sessions/{session_id}/finish",
    tags=["session"],
    summary="通知会话完成",
    description="上游图片或视频请求结束后调用。该接口幂等。",
)
async def finish_session(
    session_id: str,
    request: FinishRequest,
    _: str = Depends(verify_remote_browser_api_key),
) -> Dict[str, Any]:
    service = _require_runtime_service()
    entry = await runtime.session_registry.pop(session_id)
    if entry and entry.browser_ref is not None:
        try:
            await service.report_request_finished(entry.browser_ref)
        except Exception as e:
            debug_logger.log_warning(f"[RemoteBrowser] finish 上报失败: {e}")

    await runtime._append_completed_task(
        task_type="finish",
        status="success",
        project_id=entry.project_id if entry else None,
        action=entry.action if entry else None,
        token_id=entry.token_id if entry else None,
        session_id=session_id,
        browser_ref=str(entry.browser_ref) if entry and entry.browser_ref is not None else None,
        proxy_source=entry.proxy_source if entry else "direct",
        proxy_url=entry.proxy_url if entry else None,
        note=request.status,
    )
    return {
        "success": True,
        "found": entry is not None,
        "session_id": session_id,
        "status": request.status,
    }


@app.post(
    "/api/v1/sessions/{session_id}/error",
    tags=["session"],
    summary="通知会话失败",
    description="上游请求失败后调用。该接口幂等，会触发本地浏览器错误上报逻辑。",
)
async def error_session(
    session_id: str,
    request: ErrorRequest,
    _: str = Depends(verify_remote_browser_api_key),
) -> Dict[str, Any]:
    service = _require_runtime_service()
    entry = await runtime.session_registry.pop(session_id)
    if entry and entry.browser_ref is not None:
        try:
            await service.report_error(
                browser_ref=entry.browser_ref,
                error_reason=request.error_reason,
            )
        except Exception as e:
            debug_logger.log_warning(f"[RemoteBrowser] error 上报失败: {e}")

    await runtime._append_completed_task(
        task_type="error",
        status="failed",
        project_id=entry.project_id if entry else None,
        action=entry.action if entry else None,
        token_id=entry.token_id if entry else None,
        session_id=session_id,
        browser_ref=str(entry.browser_ref) if entry and entry.browser_ref is not None else None,
        proxy_source=entry.proxy_source if entry else "direct",
        proxy_url=entry.proxy_url if entry else None,
        error=request.error_reason,
    )
    return {
        "success": True,
        "found": entry is not None,
        "session_id": session_id,
        "error_reason": request.error_reason,
    }


@app.post(
    "/api/v1/custom-score",
    tags=["captcha"],
    summary="执行页面内分数测试",
    description="在远程浏览器内完成 token 获取和 verify 页面校验，直接返回 score 相关结果。",
)
async def custom_score(
    request: CustomScoreRequest,
    _: str = Depends(verify_remote_browser_api_key),
) -> Dict[str, Any]:
    service = _require_runtime_service()
    selected_proxy_url, selected_proxy_source = await runtime.resolve_proxy_for_token(None)
    task = await runtime._start_task(
        task_type="custom_score",
        action=request.action,
        proxy_source=selected_proxy_source,
        proxy_url=selected_proxy_url,
        extra={
            "website_url": request.website_url,
            "verify_url": request.verify_url,
        },
    )
    try:
        payload, browser_id = await service.get_custom_score(
            website_url=request.website_url,
            website_key=request.website_key,
            verify_url=request.verify_url,
            action=request.action,
            enterprise=request.enterprise,
            proxy_url_override=selected_proxy_url,
        )
        result = dict(payload or {})
        fingerprint = await service.get_fingerprint(browser_id) or {}
        actual_proxy_url = str(fingerprint.get("proxy_url") or selected_proxy_url or "").strip() or None
        actual_proxy_source = selected_proxy_source
        if actual_proxy_url and actual_proxy_source == "direct":
            actual_proxy_source = "browser_config"
        if fingerprint and "fingerprint" not in result:
            result["fingerprint"] = fingerprint
        result.setdefault("success", bool(result.get("token")))

        await runtime._finish_task(
            task.task_id,
            status="success" if result.get("success") else "failed",
            browser_ref=str(browser_id),
            proxy_source=actual_proxy_source,
            proxy_url=actual_proxy_url,
            extra={
                "verify_http_status": result.get("verify_http_status"),
                "verify_mode": result.get("verify_mode"),
                "fingerprint": fingerprint,
            },
            error=None if result.get("success") else str(result.get("message") or "custom-score failed"),
        )
        return result
    except HTTPException:
        await runtime._finish_task(task.task_id, status="failed", error="HTTPException")
        raise
    except Exception as e:
        await runtime._finish_task(
            task.task_id,
            status="failed",
            error=str(e),
            proxy_source=selected_proxy_source,
            proxy_url=selected_proxy_url,
        )
        debug_logger.log_error(f"[RemoteBrowser] custom-score 失败: {e}")
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.remote_browser_service.app:app",
        host=str(os.getenv("REMOTE_BROWSER_HOST", "0.0.0.0") or "0.0.0.0"),
        port=_env_int("REMOTE_BROWSER_PORT", default=8060, minimum=1, maximum=65535),
        reload=False,
    )
