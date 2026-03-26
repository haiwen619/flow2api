"""Token manager for Flow2API with AT auto-refresh"""
import asyncio
import base64
import httpx
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any, Dict
from ..core.database import Database
from ..core.models import Token, Project
from ..core.logger import debug_logger
from .flow_client import FlowClient
from .proxy_manager import ProxyManager


class TokenManager:
    """Token lifecycle manager with AT auto-refresh"""

    def __init__(self, db: Database, flow_client: FlowClient):
        self.db = db
        self.flow_client = flow_client
        self._lock = asyncio.Lock()
        self._project_lock = asyncio.Lock()
        self._refresh_futures: dict[int, asyncio.Task] = {}
        self._project_pool_size = 4

    def _sort_projects(self, projects: List[Project]) -> List[Project]:
        """Sort projects in a stable order for round-robin selection."""
        return sorted(projects, key=lambda project: (project.id or 0, project.project_id))

    def _normalize_project_name_base(self, project_name: Optional[str] = None) -> str:
        """Normalize a project base name for pooled creation."""
        raw_name = (project_name or "").strip()
        if raw_name:
            parts = raw_name.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].startswith("P") and parts[1][1:].isdigit():
                return parts[0]
            return raw_name
        return datetime.now().strftime("%b %d - %H:%M")

    def _build_project_name(self, pool_index: int, base_name: Optional[str] = None) -> str:
        """Build a project name for the pool."""
        normalized_base = self._normalize_project_name_base(base_name)
        return f"{normalized_base} P{pool_index}"

    async def _ensure_primary_project_binding(
        self,
        *,
        token_id: int,
        project_id: Optional[str],
        project_name: Optional[str] = None,
    ) -> Optional[Project]:
        """Ensure the primary project binding exists and points to the current token."""
        normalized_project_id = str(project_id or "").strip()
        if not normalized_project_id:
            return None

        normalized_project_name = str(project_name or "").strip() or self._build_project_name(1)
        project = Project(
            project_id=normalized_project_id,
            token_id=token_id,
            project_name=normalized_project_name,
            tool_name="PINHOLE",
        )
        project.id = await self.db.add_project(project)
        return project

    async def _create_project_for_token(
        self,
        token: Token,
        pool_index: int,
        base_name: Optional[str] = None,
    ) -> Project:
        """Create a new pooled project for a token and persist it."""
        project_name = self._build_project_name(pool_index, base_name)
        project_id = await self.flow_client.create_project(token.st, project_name)
        debug_logger.log_info(
            f"[PROJECT] Created pooled project for token {token.id}: {project_name} ({project_id})"
        )
        project = Project(
            project_id=project_id,
            token_id=token.id,
            project_name=project_name,
        )
        project.id = await self.db.add_project(project)
        return project

    def _select_next_project(self, token: Token, projects: List[Project]) -> Project:
        """Select the next project from the pool in round-robin order."""
        ordered_projects = self._sort_projects(projects)
        if not ordered_projects:
            raise ValueError("No available projects for token")

        if token.current_project_id:
            for index, project in enumerate(ordered_projects):
                if project.project_id == token.current_project_id:
                    return ordered_projects[(index + 1) % len(ordered_projects)]

        return ordered_projects[0]

    def _parse_expiry_datetime(
        self,
        expires: Optional[str],
        fallback_token: Optional[str] = None,
    ) -> Optional[datetime]:
        """解析过期时间，优先接口字段，失败时回退 JWT exp。"""
        if isinstance(expires, str) and expires.strip():
            try:
                return datetime.fromisoformat(expires.replace("Z", "+00:00"))
            except Exception:
                pass
        if fallback_token:
            return self._parse_jwt_exp(fallback_token)
        return None

    def _format_refresh_expiry(self, exp: Optional[datetime]) -> str:
        """统一格式化刷新记录中的过期时间展示。"""
        if not exp:
            return "未知"
        try:
            exp_aware = exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)
            return exp_aware.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "未知"

    def _is_cookie_invalid_need_relogin_detail(self, detail: str) -> bool:
        """判断 reAuth 失败详情是否属于 cookie 已失效、需重新自动登录。"""
        text = str(detail or "").lower()
        if not text:
            return False
        return (
            "interaction_required" in text
            or "cookie_invalid_need_relogin" in text
            or ("需要重新自动登录" in text and "cookie" in text)
        )

    def _is_auto_login_triggered_message(self, message: str) -> bool:
        """判断是否已成功触发账号池自动登录任务。"""
        text = str(message or "").strip()
        return text.startswith("已触发账号池自动登录(") or text.startswith("已委托主节点自动登录(")

    def _extract_auto_login_job_id(self, detail: str) -> str:
        text = str(detail or "").strip()
        if not text:
            return ""
        match = re.search(r"job_id=([0-9a-fA-F]{16,64})", text)
        return str(match.group(1) if match else "").strip()

    def _pending_auto_login_ttl_seconds(self) -> int:
        try:
            value = int(os.getenv("ACCOUNTPOOL_AUTO_LOGIN_PENDING_TTL_SEC", "900") or "900")
        except Exception:
            value = 900
        return max(300, value)

    def _get_auto_refresh_advance_seconds(
        self,
        token_id: int,
        at_expires: Optional[datetime],
    ) -> int:
        """获取 AT 自动刷新提前窗口，默认 1 小时。"""
        try:
            value = int(os.getenv("FLOW_AUTO_REFRESH_ADVANCE_SECONDS", "3600") or "3600")
        except Exception:
            value = 3600
        return max(300, min(4 * 3600, value))

    def _normalize_utc_datetime(self, value: Optional[datetime]) -> Optional[datetime]:
        """将 datetime 规范化为 UTC aware。"""
        if value is None:
            return None
        try:
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _get_auto_refresh_retry_cooldown_seconds(self) -> int:
        """无过期时间场景下的自动刷新重试冷却，避免每分钟重复触发。"""
        try:
            value = int(os.getenv("FLOW_AUTO_REFRESH_RETRY_COOLDOWN_SECONDS", "900") or "900")
        except Exception:
            value = 900
        return max(300, min(3600, value))

    def _has_recent_auto_refresh_attempt(
        self,
        token: Token,
        now: Optional[datetime] = None,
    ) -> bool:
        """同一自动刷新窗口内只尝试一次，避免后台巡检每分钟重复刷新。"""
        method = str(getattr(token, "last_refresh_method", "") or "").strip().upper()
        if not method.startswith("AUTO_"):
            return False

        now_aware = self._normalize_utc_datetime(now or datetime.now(timezone.utc))
        last_refresh_at = self._normalize_utc_datetime(getattr(token, "last_refresh_at", None))
        if now_aware is None or last_refresh_at is None or last_refresh_at > now_aware:
            return False

        at_expires = self._normalize_utc_datetime(getattr(token, "at_expires", None))
        token_id = getattr(token, "id", None)
        if at_expires is not None and token_id is not None:
            advance_seconds = self._get_auto_refresh_advance_seconds(
                token_id=int(token_id),
                at_expires=at_expires,
            )
            window_start = at_expires - timedelta(seconds=advance_seconds)
            return last_refresh_at >= window_start

        cooldown_seconds = self._get_auto_refresh_retry_cooldown_seconds()
        return (now_aware - last_refresh_at).total_seconds() < cooldown_seconds

    async def _trigger_accountpool_auto_login_if_enabled(
        self,
        token_id: int,
        email: Optional[str],
    ) -> str:
        """当开关启用时，按邮箱尝试触发账号池单账号自动登录（验活）一次。"""
        from ..core.config import config

        if not config.reauth_cookie_invalid_auto_login_enabled:
            return "自动登录开关未启用"

        mail = str(email or "").strip().lower()
        if not mail or "@" not in mail:
            return "邮箱为空或格式无效，无法触发账号池自动登录"

        if config.cluster_enabled and config.cluster_role == "worker":
            return await self._delegate_auto_login_to_master(token_id=token_id, email=mail)

        try:
            from .. import main as main_app
        except Exception as import_err:
            return f"账号池服务导入失败: {import_err}"

        accountpool_service = getattr(main_app, "accountpool_service", None)
        if accountpool_service is None:
            return "账号池服务未初始化"

        try:
            listed = await accountpool_service.list_accounts(
                offset=0,
                limit=200,
                search=mail,
                platform=None,
            )
            items = listed.get("items", []) if isinstance(listed, dict) else []
        except Exception as list_err:
            return f"查询账号池失败: {list_err}"

        matched: Optional[Dict[str, Any]] = None
        for item in items:
            if not isinstance(item, dict):
                continue
            display_name = str(item.get("display_name") or "").strip().lower()
            uid = str(item.get("uid") or "").strip().lower()
            account_key = str(item.get("account_key") or "").strip().lower()
            if display_name == mail or uid == mail or account_key.endswith(f":{mail}"):
                matched = item
                break

        if not matched:
            return f"账号池不存在邮箱为 {mail} 的账号，未触发自动登录"

        account_key = str(matched.get("account_key") or "").strip()
        if not account_key:
            return "账号池匹配记录缺少account_key，未触发自动登录"

        try:
            job_id = await accountpool_service.trigger_single_validate(
                account_key=account_key,
                params={
                    "external_browser": False,
                    "manual": False,
                    "timeout_sec": 300,
                    "auto_enable_token_on_sync": True,
                    "bitbrowser": True,
                    "bitbrowser_auto_delete": True,
                    # 自动登录优先复用固定测试窗口，避免服务器侧反复 createBrowser 命中窗口数上限。
                    "reuse_test_bitbrowser_id": True,
                },
            )
            debug_logger.log_info(
                f"[AT_REFRESH] Token {token_id}: 已触发账号池自动登录 account_key={account_key} job_id={job_id}"
            )
            return f"已触发账号池自动登录(account_key={account_key}, job_id={job_id})"
        except Exception as trigger_err:
            return f"触发账号池自动登录失败: {trigger_err}"

    async def _delegate_auto_login_to_master(
        self,
        *,
        token_id: int,
        email: str,
    ) -> str:
        from ..core.config import config

        def _normalize_cluster_internal_base(value: str) -> str:
            text = str(value or "").strip().rstrip("/")
            if text.lower().endswith("/manage"):
                return text[:-7].rstrip("/")
            return text

        master_base_url = _normalize_cluster_internal_base(config.cluster_master_base_url)
        cluster_key = str(config.cluster_key or "").strip()
        callback_base_url = _normalize_cluster_internal_base(config.cluster_effective_node_public_base_url)
        if not master_base_url or not cluster_key:
            return "当前子节点未配置主节点地址或 Cluster Key，无法委托主节点自动登录"
        if not callback_base_url:
            return "当前子节点未配置对外地址，主节点无法回传刷新结果"

        payload = {
            "source_node_id": config.cluster_node_id,
            "source_node_name": config.cluster_node_name,
            "source_base_url": callback_base_url,
            "worker_token_id": int(token_id),
            "email": str(email or "").strip().lower(),
        }

        endpoint = f"{master_base_url}/api/internal/cluster/token-auto-login-request"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "X-Cluster-Key": cluster_key,
                        "X-Cluster-Node-Id": config.cluster_node_id,
                        "X-Cluster-Origin-Role": config.cluster_role,
                    },
                )
        except Exception as exc:
            return f"委托主节点自动登录失败: {exc}"

        try:
            data = response.json()
        except Exception:
            data = {}

        if response.status_code >= 400:
            detail = ""
            if isinstance(data, dict):
                detail = str(data.get("detail") or data.get("message") or "").strip()
            if not detail:
                detail = str(response.text or "").strip()
            return f"委托主节点自动登录失败: {detail or f'HTTP {response.status_code}'}"

        delegation_id = str((data or {}).get("delegation_id") or "").strip()
        master_job_id = str((data or {}).get("master_job_id") or "").strip()
        account_key = str((data or {}).get("account_key") or "").strip()
        try:
            from .. import main as main_app
            cluster_mgr = getattr(main_app, "cluster_manager", None)
            if cluster_mgr and delegation_id:
                await cluster_mgr.record_delegated_auto_login_job(
                    delegation_id=delegation_id,
                    master_job_id=master_job_id or None,
                    worker_node_id=config.cluster_node_id,
                    worker_node_name=config.cluster_node_name,
                    worker_base_url=callback_base_url,
                    worker_token_id=int(token_id),
                    email=str(email or "").strip().lower(),
                    account_key=account_key or None,
                    status="delegated",
                    detail="子节点已把浏览器自动登录委托给主节点",
                )
        except Exception:
            pass
        debug_logger.log_info(
            f"[AT_REFRESH] Token {token_id}: 已委托主节点自动登录 delegation_id={delegation_id} master_job_id={master_job_id}"
        )
        parts = []
        if account_key:
            parts.append(f"account_key={account_key}")
        if master_job_id:
            parts.append(f"job_id={master_job_id}")
        if delegation_id:
            parts.append(f"delegation_id={delegation_id}")
        suffix = ", ".join(parts) if parts else "主节点已受理"
        return f"已委托主节点自动登录({suffix})"

    async def apply_cluster_auto_login_result(
        self,
        *,
        token_id: int,
        status: str,
        detail: Optional[str] = None,
        email: Optional[str] = None,
        session_token: Optional[str] = None,
        cookie: Optional[str] = None,
        cookie_file: Optional[str] = None,
        account_key: Optional[str] = None,
        delegation_id: Optional[str] = None,
        master_job_id: Optional[str] = None,
        master_node_id: Optional[str] = None,
        master_node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = await self.db.get_token(int(token_id))
        if not token:
            raise ValueError("token not found")

        status_text = str(status or "").strip().upper() or "FAILED"
        token_email = str(email or getattr(token, "email", "") or "").strip()
        st_value = str(session_token or "").strip()
        cookie_value = str(cookie or "").strip()
        cookie_file_value = str(cookie_file or "").strip()
        source_name = str(master_node_name or master_node_id or "主节点").strip() or "主节点"
        detail_parts = [str(detail or "").strip() or "主节点已返回自动登录处理结果"]
        detail_parts.append(f"来源={source_name}")
        if account_key:
            detail_parts.append(f"account_key={account_key}")
        if master_job_id:
            detail_parts.append(f"job_id={master_job_id}")
        if delegation_id:
            detail_parts.append(f"delegation_id={delegation_id}")
        detail_text = "；".join([part for part in detail_parts if part])

        if status_text == "BANNED":
            await self.db.update_token(
                int(token_id),
                is_active=False,
                ban_reason="google_account_disabled",
                banned_at=datetime.now(timezone.utc),
            )
            await self._record_refresh_event(
                int(token_id),
                "CLUSTER_MASTER_AUTO_LOGIN",
                "BANNED",
                detail_text,
            )
            return {"success": True, "status": "BANNED", "token_id": int(token_id), "email": token_email}

        if status_text == "SUCCESS":
            if st_value:
                await self.update_token(
                    token_id=int(token_id),
                    st=st_value,
                    cookie=(cookie_value or None),
                    cookie_file=(cookie_file_value or None),
                )
                at_ok = await self._do_refresh_at(int(token_id), st_value)
                await self.enable_token(int(token_id))
                await self._record_refresh_event(
                    int(token_id),
                    "CLUSTER_MASTER_AUTO_LOGIN",
                    "SUCCESS",
                    (
                        f"{detail_text}；主节点已同步新的 Session Token"
                        if at_ok
                        else f"{detail_text}；主节点已同步新的 Session Token，但本地 AT 补刷失败，后续会继续自动刷新"
                    ),
                )
                return {
                    "success": True,
                    "status": "SUCCESS",
                    "token_id": int(token_id),
                    "email": token_email,
                    "st_synced": True,
                    "at_refreshed": bool(at_ok),
                }

            if cookie_value or cookie_file_value:
                await self.update_token(
                    token_id=int(token_id),
                    cookie=(cookie_value or None),
                    cookie_file=(cookie_file_value or None),
                )
                await self.enable_token(int(token_id))
                await self._record_refresh_event(
                    int(token_id),
                    "CLUSTER_MASTER_AUTO_LOGIN_COOKIE",
                    "SUCCESS",
                    f"{detail_text}；主节点已同步新的 Cookie，会在后续请求中继续恢复会话",
                )
                return {
                    "success": True,
                    "status": "SUCCESS",
                    "token_id": int(token_id),
                    "email": token_email,
                    "cookie_synced": True,
                }

            status_text = "FAILED"
            detail_text = f"{detail_text}；主节点任务成功结束，但未返回新的 Session Token 或 Cookie"

        await self.db.update_token(
            int(token_id),
            is_active=False,
            ban_reason="cookie_invalid_need_relogin",
            banned_at=datetime.now(timezone.utc),
        )
        await self._record_refresh_event(
            int(token_id),
            "CLUSTER_MASTER_AUTO_LOGIN",
            "FAILED",
            detail_text,
        )
        return {"success": False, "status": status_text, "token_id": int(token_id), "email": token_email}

    async def _handle_worker_browser_refresh_fallback(
        self,
        *,
        token_id: int,
        token: Token,
        refresh_source: str,
        failure_detail: Optional[str] = None,
    ) -> bool:
        """On worker nodes, delegate browser-based recovery to master instead of running local RPA."""
        from ..core.config import config

        if not (config.cluster_enabled and config.cluster_role == "worker"):
            return False

        base_detail = str(failure_detail or "").strip() or "ST直刷/reAuth纯协议刷新均失败，需要浏览器登录恢复"
        trigger_msg = await self._trigger_accountpool_auto_login_if_enabled(
            token_id,
            getattr(token, "email", None),
        )
        auto_login_triggered = self._is_auto_login_triggered_message(trigger_msg)

        if auto_login_triggered:
            pending_detail = (
                f"{base_detail}；当前节点为子节点，不执行本机浏览器刷新；"
                f"{trigger_msg}；等待主节点完成自动登录后回传新的 Token/Cookie"
            )
            await self.db.update_token(
                token_id,
                is_active=False,
                ban_reason="refresh_failed",
                banned_at=datetime.now(timezone.utc),
            )
            await self._record_refresh_event(
                token_id,
                f"{refresh_source}:CLUSTER_MASTER_AUTO_LOGIN_PENDING",
                "PENDING",
                pending_detail,
            )
            debug_logger.log_info(
                f"[AT_REFRESH] Token {token_id}: 子节点已跳过本机浏览器刷新并委托主节点处理"
            )
            return True

        failed_detail = (
            f"{base_detail}；当前节点为子节点，未执行本机浏览器刷新；"
            f"{trigger_msg or '主节点自动登录委托未触发'}"
        )
        await self._record_refresh_event(
            token_id,
            f"{refresh_source}:WORKER_BROWSER_REFRESH_SKIPPED",
            "FAILED",
            failed_detail,
        )
        await self.disable_token(token_id, reason="refresh_failed")
        debug_logger.log_warning(
            f"[AT_REFRESH] Token {token_id}: 子节点无法执行浏览器刷新且未成功委托主节点 - {failed_detail}"
        )
        return True

    async def _record_refresh_event(
        self,
        token_id: int,
        method: str,
        status: str,
        detail: str,
    ):
        """记录最近一次刷新结果，供管理页直接展示。"""
        event_at = datetime.now(timezone.utc)
        detail_text = str(detail or "").strip() or "-"
        try:
            await self.db.add_token_refresh_history(
                int(token_id),
                method=method,
                status=status,
                detail=(detail_text[:4000] or "-"),
                created_at=event_at,
            )
            await self.db.update_token(
                token_id,
                last_refresh_at=event_at,
                last_refresh_method=method,
                last_refresh_status=status,
                last_refresh_detail=(detail_text[:500] or "-"),
            )
        except Exception as log_err:
            debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: 写入刷新记录失败 - {log_err}")

    async def reconcile_pending_auto_login_status(self, token_id: int):
        """收敛长时间未完成的账号池自动登录 pending 状态，避免界面永久显示执行中。"""
        token = await self.db.get_token(token_id)
        if not token:
            return None

        status = str(getattr(token, "last_refresh_status", "") or "").strip().upper()
        detail = str(getattr(token, "last_refresh_detail", "") or "").strip()
        method = str(getattr(token, "last_refresh_method", "") or "").strip() or "ACCOUNTPOOL_AUTO_LOGIN"
        if status not in {"PENDING", "RUNNING", "IN_PROGRESS"}:
            return token
        if "账号池自动登录" not in detail and "AUTO_LOGIN" not in method.upper():
            return token

        last_refresh_at = getattr(token, "last_refresh_at", None)
        if last_refresh_at is None:
            return token

        now = datetime.now(timezone.utc)
        refresh_at_aware = (
            last_refresh_at
            if getattr(last_refresh_at, "tzinfo", None)
            else last_refresh_at.replace(tzinfo=timezone.utc)
        )
        age_seconds = max(0, int((now - refresh_at_aware).total_seconds()))
        ttl_seconds = self._pending_auto_login_ttl_seconds()
        if age_seconds < ttl_seconds:
            return token

        job_id = self._extract_auto_login_job_id(detail)
        failed_detail = (
            f"账号池自动登录已超过 {ttl_seconds} 秒仍未完成，判定为超时；"
            "请检查RPA执行日志或重新触发自动登录"
        )

        try:
            from .. import main as main_app
            accountpool_service = getattr(main_app, "accountpool_service", None)
        except Exception:
            accountpool_service = None

        if accountpool_service is not None and job_id:
            try:
                job = accountpool_service.get_job_safe(job_id=job_id)
                job_status = str(job.get("status") or "").strip().lower()
                job_error = str(job.get("error") or "").strip()
                if job_status in {"queued", "running"}:
                    failed_detail = (
                        f"账号池自动登录任务超时未完成(status={job_status}, age={age_seconds}s)；"
                        "请检查RPA是否卡住或浏览器是否未正常退出"
                    )
                elif job_status == "failed":
                    failed_detail = (
                        f"账号池自动登录任务已失败(status=failed)"
                        f"{'：' + job_error if job_error else ''}"
                    )
                elif job_status == "cancelled":
                    failed_detail = "账号池自动登录任务已取消，未恢复Token活跃状态"
                elif job_status == "success":
                    failed_detail = "账号池自动登录任务已完成，但未同步恢复Token活跃状态"
                else:
                    failed_detail = (
                        f"账号池自动登录任务状态未知(status={job_status or 'unknown'})，"
                        "未恢复Token活跃状态"
                    )
            except KeyError:
                failed_detail = (
                    f"服务重启丢任务：账号池自动登录任务记录不存在(job_id={job_id})，"
                    "可能服务已重启或内存中的任务记录被清理；未恢复Token活跃状态"
                )
            except Exception as job_err:
                failed_detail = (
                    f"账号池自动登录状态检查失败：{job_err}；"
                    "未恢复Token活跃状态"
                )

        await self._record_refresh_event(
            token_id,
            method,
            "FAILED",
            failed_detail,
        )
        return await self.db.get_token(token_id)

    # ========== Token CRUD ==========

    async def get_all_tokens(self) -> List[Token]:
        """Get all tokens"""
        return await self.db.get_all_tokens()

    async def get_active_tokens(self) -> List[Token]:
        """Get all active tokens"""
        return await self.db.get_active_tokens()

    async def get_token(self, token_id: int) -> Optional[Token]:
        """Get token by ID"""
        return await self.db.get_token(token_id)

    async def delete_token(self, token_id: int):
        """Delete token"""
        await self.db.delete_token(token_id)

    async def delete_all_tokens(self) -> int:
        """Delete all tokens"""
        return await self.db.delete_all_tokens()

    async def enable_token(self, token_id: int):
        """Enable a token and reset error count"""
        # Enable the token
        await self.db.update_token(
            token_id,
            is_active=True,
            ban_reason=None,
            banned_at=None,
        )
        # Reset error count when enabling (only reset total error_count, keep today_error_count)
        await self.db.reset_error_count(token_id)

    async def disable_token(
        self,
        token_id: int,
        *,
        reason: Optional[str] = None,
        disabled_at: Optional[datetime] = None,
    ):
        """Disable a token and optionally persist a structured reason."""
        update_fields: Dict[str, Any] = {"is_active": False}
        if reason is not None:
            update_fields["ban_reason"] = reason
            update_fields["banned_at"] = disabled_at or datetime.now(timezone.utc)
        elif disabled_at is not None:
            update_fields["banned_at"] = disabled_at
        await self.db.update_token(token_id, **update_fields)

    # ========== Token添加 (支持Project创建) ==========

    async def add_token(
        self,
        st: str,
        cookie: Optional[str] = None,
        cookie_file: Optional[str] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        remark: Optional[str] = None,
        image_enabled: bool = True,
        video_enabled: bool = True,
        image_concurrency: int = -1,
        video_concurrency: int = -1,
        captcha_proxy_url: Optional[str] = None,
        resolved_at: Optional[str] = None,
        resolved_at_expires: Optional[datetime] = None,
        resolved_email: Optional[str] = None,
        resolved_name: Optional[str] = None,
    ) -> Token:
        """Add a new token

        Args:
            st: Session Token (必需)
            cookie: 完整 Cookie Header（可选）
            cookie_file: Google 域名 Cookie Header（可选）
            project_id: 项目ID (可选,如果提供则直接使用,不创建新项目)
            project_name: 项目名称 (可选,如果不提供则自动生成)
            remark: 备注
            image_enabled: 是否启用图片生成
            video_enabled: 是否启用视频生成
            image_concurrency: 图片并发限制
            video_concurrency: 视频并发限制
            captcha_proxy_url: token级浏览器打码代理（可选，优先于全局）
            resolved_at: 已预解析的 AT（导入场景复用，避免重复 ST->AT）
            resolved_at_expires: 已预解析的 AT 过期时间
            resolved_email: 已预解析的邮箱
            resolved_name: 已预解析的名称

        Returns:
            Token object
        """
        # Step 1: 检查ST是否已存在
        existing_token = await self.db.get_token_by_st(st)
        if existing_token:
            raise ValueError(f"Token ??????: {existing_token.email}?")

        # Step 2: 使用ST转换AT，或复用导入阶段已解析出的最新结果。
        try:
            email = str(resolved_email or "").strip()
            at = str(resolved_at or "").strip() or None
            name = str(resolved_name or "").strip()
            at_expires = resolved_at_expires

            if not email:
                debug_logger.log_info(f"[ADD_TOKEN] Converting ST to AT...")
                result = await self.flow_client.st_to_at(st)
                at = result["access_token"]
                expires = result.get("expires")
                user_info = result.get("user", {})
                email = str(user_info.get("email", "") or "").strip()
                name = str(user_info.get("name", "") or "").strip()
                at_expires = self._parse_expiry_datetime(expires, fallback_token=at)
            else:
                debug_logger.log_info(f"[ADD_TOKEN] Using pre-resolved AT/expires from import pipeline for {email}")

            if not name:
                name = email.split("@")[0] if email else ""

        except Exception as e:
            raise ValueError(f"ST?AT??: {str(e)}")

        # Step 3: 查询余额
        credits = 0
        user_paygate_tier = None
        if at:
            try:
                credits_result = await self.flow_client.get_credits(at)
                credits = credits_result.get("credits", 0)
                user_paygate_tier = credits_result.get("userPaygateTier")
            except Exception:
                credits = 0
                user_paygate_tier = None

        base_project_name = self._normalize_project_name_base(project_name)
        pooled_projects: List[Project] = []

        if project_id:
            first_project_name = self._build_project_name(1, base_project_name)
            debug_logger.log_info(f"[ADD_TOKEN] Using provided project_id as pooled project #1: {project_id}")
            pooled_projects.append(Project(
                project_id=project_id,
                token_id=0,
                project_name=first_project_name,
                tool_name="PINHOLE"
            ))
        else:
            try:
                first_project_name = self._build_project_name(1, base_project_name)
                first_project_id = await self.flow_client.create_project(st, first_project_name)
                debug_logger.log_info(f"[ADD_TOKEN] Created pooled project #1: {first_project_name} (ID: {first_project_id})")
                pooled_projects.append(Project(
                    project_id=first_project_id,
                    token_id=0,
                    project_name=first_project_name,
                    tool_name="PINHOLE"
                ))
            except Exception as e:
                raise ValueError(f"??????: {str(e)}")

        token = Token(
            st=st,
            cookie=cookie,
            cookie_file=cookie_file,
            at=at,
            at_expires=at_expires,
            email=email,
            name=name,
            remark=remark,
            is_active=True,
            credits=credits,
            user_paygate_tier=user_paygate_tier,
            current_project_id=pooled_projects[0].project_id,
            current_project_name=pooled_projects[0].project_name,
            image_enabled=image_enabled,
            video_enabled=video_enabled,
            image_concurrency=image_concurrency,
            video_concurrency=video_concurrency,
            captcha_proxy_url=captcha_proxy_url
        )

        token_id = await self.db.add_token(token)
        token.id = token_id

        pooled_projects[0].token_id = token_id
        pooled_projects[0] = await self._ensure_primary_project_binding(
            token_id=token_id,
            project_id=pooled_projects[0].project_id,
            project_name=pooled_projects[0].project_name,
        ) or pooled_projects[0]

        while len(pooled_projects) < self._project_pool_size:
            try:
                new_project = await self._create_project_for_token(token, len(pooled_projects) + 1, base_project_name)
                pooled_projects.append(new_project)
            except Exception as exc:
                debug_logger.log_warning(
                    f"[ADD_TOKEN] Token {token_id} primary import succeeded, but pooled project P{len(pooled_projects) + 1} creation failed: {exc}"
                )
                break

        debug_logger.log_info(
            f"[ADD_TOKEN] Token added successfully (ID: {token_id}, Email: {email}, pooled_projects={len(pooled_projects)})"
        )
        return token
    async def update_token(
        self,
        token_id: int,
        st: Optional[str] = None,
        cookie: Optional[str] = None,
        cookie_file: Optional[str] = None,
        at: Optional[str] = None,
        at_expires: Optional[datetime] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        remark: Optional[str] = None,
        image_enabled: Optional[bool] = None,
        video_enabled: Optional[bool] = None,
        image_concurrency: Optional[int] = None,
        video_concurrency: Optional[int] = None,
        captcha_proxy_url: Optional[str] = None
    ):
        """Update token (支持修改project_id和project_name)

        当用户编辑保存token时，如果token未过期，自动清空429禁用状态
        """
        update_fields = {}

        if st is not None:
            update_fields["st"] = st
        if cookie is not None:
            update_fields["cookie"] = cookie
        if cookie_file is not None:
            update_fields["cookie_file"] = cookie_file
        if at is not None:
            update_fields["at"] = at
        if at_expires is not None:
            update_fields["at_expires"] = at_expires
        if project_id is not None:
            update_fields["current_project_id"] = project_id
        if project_name is not None:
            update_fields["current_project_name"] = project_name
        if remark is not None:
            update_fields["remark"] = remark
        if image_enabled is not None:
            update_fields["image_enabled"] = image_enabled
        if video_enabled is not None:
            update_fields["video_enabled"] = video_enabled
        if image_concurrency is not None:
            update_fields["image_concurrency"] = image_concurrency
        if video_concurrency is not None:
            update_fields["video_concurrency"] = video_concurrency
        if captcha_proxy_url is not None:
            update_fields["captcha_proxy_url"] = captcha_proxy_url

        # 检查token是否因429被禁用，如果是且未过期，则清空429状态
        token = await self.db.get_token(token_id)
        if token and token.ban_reason == "429_rate_limit":
            # 检查token是否过期
            is_expired = False
            if token.at_expires:
                now = datetime.now(timezone.utc)
                if token.at_expires.tzinfo is None:
                    at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
                else:
                    at_expires_aware = token.at_expires
                is_expired = at_expires_aware <= now

            # 如果未过期，清空429禁用状态
            if not is_expired:
                debug_logger.log_info(f"[UPDATE_TOKEN] Token {token_id} 编辑保存，清空429禁用状态")
                update_fields["ban_reason"] = None
                update_fields["banned_at"] = None

        if update_fields:
            await self.db.update_token(token_id, **update_fields)
            if project_id is not None:
                await self._ensure_primary_project_binding(
                    token_id=token_id,
                    project_id=project_id,
                    project_name=project_name,
                )

    # ========== AT自动刷新逻辑 (核心) ==========

    def _should_refresh_at(self, token: Token) -> bool:
        """根据当前 token 快照判断是否需要刷新 AT。"""
        token_id = getattr(token, "id", "unknown")

        if not token.at:
            debug_logger.log_info(f"[AT_CHECK] Token {token_id}: AT不存在,需要刷新")
            return True

        if not token.at_expires:
            debug_logger.log_info(f"[AT_CHECK] Token {token_id}: AT过期时间未知,尝试刷新")
            return True

        now = datetime.now(timezone.utc)
        if token.at_expires.tzinfo is None:
            at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
        else:
            at_expires_aware = token.at_expires

        time_until_expiry = at_expires_aware - now
        advance_seconds = self._get_auto_refresh_advance_seconds(
            token_id=int(token.id),
            at_expires=at_expires_aware,
        )
        if time_until_expiry.total_seconds() < advance_seconds:
            debug_logger.log_info(
                f"[AT_CHECK] Token {token_id}: AT即将过期 "
                f"(剩余 {time_until_expiry.total_seconds():.0f} 秒, "
                f"触发阈值 {advance_seconds / 3600:.2f} 小时),需要刷新"
            )
            return True

        return False

    async def ensure_valid_token(self, token: Optional[Token]) -> Optional[Token]:
        """确保 token 的 AT 可用，并在必要时返回刷新后的最新对象。"""
        if not token:
            return None

        if not self._should_refresh_at(token):
            return token

        token_id = token.id
        refresh_source = "AUTO_NEAR_EXPIRY"
        if not token.at:
            refresh_source = "AUTO_MISSING_AT"
        elif not token.at_expires:
            refresh_source = "AUTO_NO_EXPIRY"

        reauth_attempted = False
        if token.at and token.at_expires:
            from ..core.config import config

            if config.enable_reauth_refresh:
                reauth_attempted = True
                debug_logger.log_info(
                    f"[AT_CHECK] Token {token_id}: 先尝试 reAuth 纯协议刷新会话..."
                )
                if await self.refresh_cookie_via_reauth(token_id, refresh_source="AUTO_REAUTH"):
                    debug_logger.log_info(
                        f"[AT_CHECK] Token {token_id}: reAuth 纯协议刷新成功"
                    )
                    return await self.db.get_token(token_id)

                latest = await self.db.get_token(token_id)
                last_detail = str(getattr(latest, "last_refresh_detail", "") or "")
                if self._is_cookie_invalid_need_relogin_detail(last_detail):
                    debug_logger.log_warning(f"[AT_CHECK] Token {token_id}: {last_detail}")
                    return None

                debug_logger.log_warning(
                    f"[AT_CHECK] Token {token_id}: reAuth 纯协议刷新失败，回退 AT 直刷链路"
                )

        if not await self._refresh_at(
            token_id,
            skip_reauth_fallback=reauth_attempted,
            refresh_source=refresh_source,
        ):
            return None

        return await self.db.get_token(token_id)

    async def is_at_valid(self, token_id: int, token: Optional[Token] = None) -> bool:
        """检查 AT 是否有效，如有必要则自动刷新。"""
        token_obj = token if token and token.id == token_id else await self.db.get_token(token_id)
        if not token_obj:
            return False

        valid_token = await self.ensure_valid_token(token_obj)
        return valid_token is not None


    async def _refresh_at(
        self,
        token_id: int,
        skip_reauth_fallback: bool = False,
        refresh_source: str = "MANUAL_AT",
    ) -> bool:
        """Coalesce concurrent AT refresh calls for the same token."""
        existing_task = self._refresh_futures.get(token_id)
        if existing_task:
            return await existing_task

        async def runner() -> bool:
            try:
                return await self._refresh_at_inner(
                    token_id,
                    skip_reauth_fallback=skip_reauth_fallback,
                    refresh_source=refresh_source,
                )
            finally:
                current = self._refresh_futures.get(token_id)
                if current is task:
                    self._refresh_futures.pop(token_id, None)

        task = asyncio.create_task(runner())
        self._refresh_futures[token_id] = task
        return await task

    async def _refresh_at_inner(
        self,
        token_id: int,
        skip_reauth_fallback: bool = False,
        refresh_source: str = "MANUAL_AT",
    ) -> bool:
        """执行一次真实的 AT 刷新流程。"""
        async with self._lock:
            token = await self.db.get_token(token_id)
            if not token:
                return False

            # # 第一次尝试刷新 AT
            result = await self._do_refresh_at(token_id, token.st)
            if result:
                updated = await self.db.get_token(token_id)
                await self._record_refresh_event(
                    token_id,
                    f"{refresh_source}:ST_TO_AT",
                    "SUCCESS",
                    f"AT刷新成功，过期时间={self._format_refresh_expiry(updated.at_expires if updated else None)}",
                )
                return True

            # AT 刷新失败，可选尝试 HTTP reAuth 恢复可用 AT（可能先拿到 ST 再换 AT，或直接拿到可用 AT）
            from ..core.config import config
            reauth_detail = ""
            if config.enable_reauth_refresh and not skip_reauth_fallback:
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 第一次 AT 刷新失败，尝试 reAuth 恢复 AT...")
                reauth_result, reauth_detail = await self._try_refresh_at_via_reauth(token_id, token)
                if reauth_result:
                    debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: reAuth 恢复 AT 成功")
                    updated = await self.db.get_token(token_id)
                    await self._record_refresh_event(
                        token_id,
                        f"{refresh_source}:REAUTH_FALLBACK",
                        "SUCCESS",
                        f"reAuth恢复AT成功，过期时间={self._format_refresh_expiry(updated.at_expires if updated else None)}",
                    )
                    return True
                if reauth_detail:
                    debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: reAuth 恢复失败详情: {reauth_detail}")
                    if self._is_cookie_invalid_need_relogin_detail(reauth_detail):
                        invalid_detail = (
                            "reAuth命中interaction_required，判定当前记录的Cookie已失效，"
                            "已标记为失效并停用，后续自动刷新将跳过，需重新自动登录恢复"
                        )
                        auto_login_msg = await self._trigger_accountpool_auto_login_if_enabled(
                            token_id,
                            getattr(token, "email", None),
                        )
                        auto_login_triggered = self._is_auto_login_triggered_message(auto_login_msg)
                        if auto_login_msg:
                            invalid_detail = f"{invalid_detail}；{auto_login_msg}"
                        if auto_login_triggered:
                            invalid_detail = f"{invalid_detail}；账号池自动登录执行中，请稍候自动恢复"
                        await self.db.update_token(
                            token_id,
                            is_active=False,
                            ban_reason="cookie_invalid_need_relogin",
                            banned_at=datetime.now(timezone.utc),
                        )
                        await self._record_refresh_event(
                            token_id,
                            f"{refresh_source}:REAUTH_INVALID_COOKIE_AUTO_LOGIN_PENDING"
                            if auto_login_triggered
                            else f"{refresh_source}:REAUTH_INVALID_COOKIE",
                            "PENDING" if auto_login_triggered else "FAILED",
                            invalid_detail,
                        )
                        debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: {invalid_detail}")
                        return False
            elif skip_reauth_fallback:
                debug_logger.log_info(
                    f"[AT_REFRESH] Token {token_id}: 已在外层执行过 reAuth，跳过本轮 reAuth fallback"
                )
            else:
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: reAuth 恢复已禁用（flow.enable_reauth_refresh=false），跳过")

            worker_fallback_detail = "ST直刷/reAuth恢复均失败，需要浏览器登录恢复"
            if reauth_detail:
                worker_fallback_detail = f"{worker_fallback_detail}；reAuth详情={str(reauth_detail)[:240]}"
            if await self._handle_worker_browser_refresh_fallback(
                token_id=token_id,
                token=token,
                refresh_source=refresh_source,
                failure_detail=worker_fallback_detail,
            ):
                return False

            # 继续尝试 personal 模式浏览器刷新 ST
            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 尝试浏览器刷新 ST...")
            new_st = await self._try_refresh_st(token_id, token)
            if new_st:
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 浏览器刷新 ST 成功，重试 AT 刷新...")
                result = await self._do_refresh_at(token_id, new_st)
                if result:
                    updated = await self.db.get_token(token_id)
                    await self._record_refresh_event(
                        token_id,
                        f"{refresh_source}:BROWSER_ST",
                        "SUCCESS",
                        f"浏览器刷新ST后AT刷新成功，过期时间={self._format_refresh_expiry(updated.at_expires if updated else None)}",
                    )
                    return True

            # 所有刷新尝试都失败，禁用 Token
            debug_logger.log_error(f"[AT_REFRESH] Token {token_id}: 所有刷新尝试失败，禁用 Token")
            failed_detail = "ST直刷/reAuth恢复/浏览器刷新均失败，Token已自动禁用"
            if reauth_detail:
                failed_detail = f"{failed_detail}；reAuth详情={str(reauth_detail)[:240]}"
            await self._record_refresh_event(
                token_id,
                f"{refresh_source}:ALL",
                "FAILED",
                failed_detail,
            )
            await self.disable_token(token_id, reason="refresh_failed")
            return False

    async def refresh_cookie_via_reauth(
        self,
        token_id: int,
        refresh_source: str = "MANUAL_COOKIE",
    ) -> bool:
        """仅通过 reAuth 刷新 Cookie/会话，不执行首次 ST->AT 直刷。

        用于手动“刷新Cookie”按钮：
        - 直接走 HTTP reAuth
        - 跳过 `_refresh_at` 中第一次 `_do_refresh_at` 尝试
        """
        async with self._lock:
            token = await self.db.get_token(token_id)
            if not token:
                return False

            old_cookie = str(token.cookie or "").strip()
            debug_logger.log_info(f"[REAUTH_ONLY] Token {token_id}: 开始仅 reAuth 刷新 Cookie...")
            success, reauth_detail = await self._try_refresh_at_via_reauth(token_id, token)
            if not success:
                debug_logger.log_warning(f"[REAUTH_ONLY] Token {token_id}: reAuth 刷新失败")
                detail_msg = str(reauth_detail or "reAuth刷新Cookie失败（未恢复可用AT）")
                auto_login_triggered = False
                if self._is_cookie_invalid_need_relogin_detail(detail_msg):
                    detail_msg = (
                        "reAuth命中interaction_required，判定当前记录的Cookie已失效，"
                        "已标记为失效并停用，后续自动刷新将跳过，需重新自动登录恢复"
                    )
                    auto_login_msg = await self._trigger_accountpool_auto_login_if_enabled(
                        token_id,
                        getattr(token, "email", None),
                    )
                    auto_login_triggered = self._is_auto_login_triggered_message(auto_login_msg)
                    if auto_login_msg:
                        detail_msg = f"{detail_msg}；{auto_login_msg}"
                    if auto_login_triggered:
                        detail_msg = f"{detail_msg}；账号池自动登录执行中，请稍候自动恢复"
                    await self.db.update_token(
                        token_id,
                        is_active=False,
                        ban_reason="cookie_invalid_need_relogin",
                        banned_at=datetime.now(timezone.utc),
                    )
                await self._record_refresh_event(
                    token_id,
                    f"{refresh_source}:REAUTH_ONLY_AUTO_LOGIN_PENDING"
                    if auto_login_triggered
                    else f"{refresh_source}:REAUTH_ONLY",
                    "PENDING" if auto_login_triggered else "FAILED",
                    detail_msg,
                )
                return False

            updated = await self.db.get_token(token_id)
            new_cookie = str((updated.cookie if updated else "") or "").strip()
            cookie_changed = bool(new_cookie and new_cookie != old_cookie)
            reactivated = False
            if updated and str(getattr(updated, "ban_reason", "") or "") == "cookie_invalid_need_relogin":
                await self.enable_token(token_id)
                updated = await self.db.get_token(token_id)
                reactivated = True
            debug_logger.log_info(
                f"[REAUTH_ONLY] Token {token_id}: reAuth 刷新成功 cookie_changed={cookie_changed} reactivated={reactivated}"
            )
            await self._record_refresh_event(
                token_id,
                f"{refresh_source}:REAUTH_ONLY",
                "SUCCESS",
                f"reAuth刷新成功 cookie_changed={cookie_changed} reactivated={reactivated}，过期时间={self._format_refresh_expiry(updated.at_expires if updated else None)}",
            )
            return True

    async def _do_refresh_at(self, token_id: int, st: str, st_raw: Optional[str] = None) -> bool:
        """执行 AT 刷新的核心逻辑

        Args:
            token_id: Token ID
            st: Session Token

        Returns:
            True if refresh successful AND AT is valid, False otherwise
        """
        try:
            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 开始刷新AT...")

            # 兼容输入：若误传入 Cookie/Set-Cookie 文本（含 Path/Expires 等属性），先提取纯 ST。
            st_input = str(st or "")
            if "__Secure-next-auth.session-token" in st_input or ";" in st_input:
                try:
                    from reAuth.refresh_cookie_before_relogin import extract_session_token_from_cookie_header
                    extracted = extract_session_token_from_cookie_header(st_input)
                    if extracted:
                        debug_logger.log_info(
                            f"[AT_REFRESH] Token {token_id}: 检测到Cookie/Set-Cookie格式输入，已提取纯ST（输入长度={len(st_input)} -> ST长度={len(extracted)}）"
                        )
                        st = extracted
                except Exception as norm_err:
                    debug_logger.log_warning(
                        f"[AT_REFRESH] Token {token_id}: ST 输入规范化失败，继续使用原值: {norm_err}"
                    )
            
            # 临时调试位：如需强制使用网页最新 ST 进行测试，可在此处填入。
            # 留空字符串时不生效，继续使用函数入参 st。
            debug_st_override = ""
            st_for_refresh = (str(debug_st_override).strip() or st)
            if str(debug_st_override).strip():
                debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: 使用临时调试 ST 覆盖入参 ST")
            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 当前ST长度={len(str(st_for_refresh or ''))}")
            if st_raw:
                print(f"  - 当前 ST(原文): {st_raw}")
            print(f"  - 当前 ST: {st_for_refresh}")
            # 使用ST转AT
            result = await self.flow_client.st_to_at(st_for_refresh)
            new_at = result["access_token"]
            expires = result.get("expires")

            # 解析过期时间
            new_at_expires = self._parse_expiry_datetime(expires, fallback_token=new_at)

            # 更新数据库
            await self.db.update_token(
                token_id,
                at=new_at,
                at_expires=new_at_expires
            )

            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: AT刷新成功")
            debug_logger.log_info(f"  - 新过期时间(UTC): {new_at_expires}")
            beijing_expires_text = None
            if new_at_expires is not None:
                try:
                    beijing_tz = timezone(timedelta(hours=8))
                    if new_at_expires.tzinfo is None:
                        exp_aware = new_at_expires.replace(tzinfo=timezone.utc)
                    else:
                        exp_aware = new_at_expires
                    beijing_expires_text = exp_aware.astimezone(beijing_tz).isoformat()
                except Exception:
                    beijing_expires_text = None
            debug_logger.log_info(f"  - 新过期时间(北京时间): {beijing_expires_text or '未知'}")

            # 验证 AT 有效性：通过 get_credits 测试
            try:
                print(f"  - 验证新 AT 是否有效...")
                print(f"  - 新 AT: {new_at}")
                credits_result = await self.flow_client.get_credits(new_at)
                await self.db.update_token(
                    token_id,
                    credits=credits_result.get("credits", 0),
                    user_paygate_tier=credits_result.get("userPaygateTier"),
                )
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: AT 验证成功（余额: {credits_result.get('credits', 0)}）")
                return True
            except Exception as verify_err:
                # AT 验证失败（可能返回 401），说明 ST 已过期
                error_msg = str(verify_err)
                if "401" in error_msg or "UNAUTHENTICATED" in error_msg:
                    debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: AT 验证失败 (401)，ST 可能已过期")
                    return False
                else:
                    # 其他错误（如网络问题），仍视为成功
                    debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: AT 验证时发生非认证错误: {error_msg}")
                    return True

        except Exception as e:
            debug_logger.log_error(f"[AT_REFRESH] Token {token_id}: AT刷新失败 - {str(e)}")
            return False

    def _parse_jwt_exp(self, token: str) -> Optional[datetime]:
        """从 JWT 中提取 exp 并转为 UTC datetime"""
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload = parts[1]
            payload += "=" * ((4 - len(payload) % 4) % 4)
            decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
            import json
            payload_obj = json.loads(decoded)
            exp = payload_obj.get("exp")
            if not exp:
                return None
            return datetime.fromtimestamp(int(exp), tz=timezone.utc)
        except Exception:
            return None

    async def _accept_direct_at_if_valid(self, token_id: int, candidate_at: str) -> bool:
        """将候选 token 作为 AT 直接验证并写回"""
        try:
            credits_result = await self.flow_client.get_credits(candidate_at)
            at_expires = self._parse_jwt_exp(candidate_at)
            await self.db.update_token(
                token_id,
                at=candidate_at,
                at_expires=at_expires,
                credits=credits_result.get("credits", 0),
            )
            debug_logger.log_info(
                f"[REAUTH_AT] Token {token_id}: 候选 token 可作为 AT 使用（余额: {credits_result.get('credits', 0)}）"
            )
            return True
        except Exception as e:
            debug_logger.log_warning(f"[REAUTH_AT] Token {token_id}: 候选 token 作为 AT 验证失败 - {str(e)}")
            return False

    async def _try_refresh_at_via_reauth(self, token_id: int, token) -> tuple[bool, str]:
        """尝试通过 reAuth HTTP 流程恢复 AT（优先按 ST/RT->AT，兼容 AT 直验）"""
        try:
            if not token.current_project_id:
                debug_logger.log_warning(f"[REAUTH_ST] Token {token_id} 没有 project_id，无法执行 reAuth")
                return False, "缺少 project_id，无法执行 reAuth"

            debug_logger.log_info(f"[REAUTH_AT] Token {token_id}: 开始执行 reAuth，尝试恢复 AT...")

            # 延迟导入，避免主流程无谓依赖
            from reAuth.refresh_cookie_before_relogin import (
                ReAuthAccount,
                refresh_cookie_before_relogin,
                _st_to_at_via_auth_session,
                _verify_at_via_get_credits,
                REAUTH_COOKIE_INVALID_NEED_RELOGIN,
            )

            # 优先使用完整 Cookie（由账号池自动化同步），没有则回退最小 ST Cookie
            initial_cookie_header = str(token.cookie or "").strip() or f"__Secure-next-auth.session-token={token.st}"
            initial_cookie_file_header = str(token.cookie_file or "").strip()
            old_st_value = str(token.st or "").strip()

            proxy_url = None
            try:
                proxy_config = await self.db.get_proxy_config()
                if proxy_config and proxy_config.enabled and proxy_config.proxy_url:
                    proxy_url = proxy_config.proxy_url
            except Exception as proxy_err:
                debug_logger.log_warning(f"[REAUTH_AT] Token {token_id}: 读取代理配置失败，继续直连: {proxy_err}")

            seed_cookie = initial_cookie_header
            seed_cookie_file = initial_cookie_file_header
            last_candidate_token = ""
            candidate_source = "__Secure-next-auth.session-token"
            failure_detail = "reAuth流程未恢复可用AT"
            # 暂时收敛为单次纯协议尝试，便于聚焦六步链路调试。
            strategies = [
                ("default-pass1", False),
            ]

            for attempt_idx, (strategy_name, strict_mode) in enumerate(strategies, start=1):
                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: 协议刷新尝试 {attempt_idx}/{len(strategies)} strategy={strategy_name} strict={strict_mode}"
                )

                def _run_reauth_once(
                    cookie_seed: str = seed_cookie,
                    strict_value: bool = strict_mode,
                    cookie_file_seed: str = seed_cookie_file,
                ) -> tuple[str, str, str]:
                    account = ReAuthAccount(
                        project_id=token.current_project_id,
                        cookie=cookie_seed,
                        cookie_file=(cookie_file_seed or None),
                    )
                    refresh_cookie_before_relogin(
                        account=account,
                        timeout=30,
                        proxy_url=proxy_url,
                        strict_java_flow=strict_value,
                    )
                    # 使用 account 内部最终状态，避免依赖 refresh 函数返回值格式。
                    refreshed_cookie_value = str(getattr(account, "cookie", "") or "").strip()
                    refreshed_st_value = str(getattr(account, "final_session_token", "") or "").strip()
                    refreshed_st_raw = str(getattr(account, "final_session_set_cookie_raw", "") or "").strip()
                    return refreshed_cookie_value, refreshed_st_value, refreshed_st_raw

                try:
                    refreshed_cookie, refreshed_st, refreshed_st_raw = await asyncio.to_thread(_run_reauth_once)
                except Exception as reauth_err:
                    err_text = str(reauth_err or "")
                    err_text_l = err_text.lower()
                    if (REAUTH_COOKIE_INVALID_NEED_RELOGIN in err_text) or ("interaction_required" in err_text_l):
                        failure_detail = (
                            "reAuth检测到step4返回 interaction_required，判定当前记录的cookie已失效或受其他登录状态影响，"
                            "需要重新自动登录"
                        )
                        debug_logger.log_warning(
                            f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} {failure_detail}"
                        )
                        return False, failure_detail
                    failure_detail = f"reAuth流程异常（{strategy_name}）: {err_text}"
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 执行异常 - {err_text}"
                    )
                    continue
                print(f"  - reAuth[{strategy_name}] 返回 cookie: {refreshed_cookie}")
                print(f"  - reAuth[{strategy_name}] 提取 ST: {refreshed_st_raw or refreshed_st}")
                candidate_token = refreshed_st
                if candidate_token:
                    last_candidate_token = candidate_token

                if refreshed_st_raw and ";" in refreshed_st_raw:
                    debug_logger.log_info(
                        f"[REAUTH_AT] Token {token_id}: 提取到Set-Cookie原文（含Path/Expires属性），ST->AT将使用纯ST值"
                    )
                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: reAuth返回cookie长度={len(str(refreshed_cookie or ''))}, 提取ST长度={len(str(refreshed_st or ''))}, 提取ST原文长度={len(str(refreshed_st_raw or ''))}"
                )
                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: reAuth前后ST对比 old_len={len(old_st_value)} new_len={len(str(refreshed_st or ''))} changed={bool(refreshed_st and refreshed_st != old_st_value)}"
                )

                if refreshed_cookie:
                    # 保存 reAuth 返回的完整 cookie，供下次 reAuth/排障复用
                    await self.db.update_token(token_id, cookie=refreshed_cookie)
                    seed_cookie = refreshed_cookie

                if not candidate_token:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 未提取到可用 session token，继续下一策略"
                    )
                    continue

                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 候选 token 来源={candidate_source}"
                )

                if refreshed_st and refreshed_st != old_st_value:
                    await self.db.update_token(token_id, st=refreshed_st)
                    debug_logger.log_info(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 已更新 ST，尝试 ST->AT"
                    )
                    old_st_value = refreshed_st
                elif refreshed_st:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 提取到的 ST 与旧值相同，继续尝试恢复 AT"
                    )
                else:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 未提取到 ST，仅使用候选 token 继续恢复 AT"
                    )

                # 当前策略下优先使用提取到的 ST，未提取到时回退候选 token。
                fallback_st = refreshed_st or candidate_token

                # 路径0：先用 reAuth 同代理链路直接验证一次（避免 reAuth 与 flow_client 请求画像差异）。
                try:
                    st_to_at_direct = _st_to_at_via_auth_session(
                        fallback_st,
                        proxy_url=proxy_url,
                        connect_timeout=10,
                        timeout=30,
                    )
                    if st_to_at_direct.get("success"):
                        direct_at = str(st_to_at_direct.get("access_token") or "").strip()
                        direct_expires = st_to_at_direct.get("expires")
                        verify_direct = _verify_at_via_get_credits(
                            direct_at,
                            proxy_url=proxy_url,
                            connect_timeout=10,
                            timeout=30,
                        )
                        if verify_direct.get("success"):
                            direct_at_expires = self._parse_expiry_datetime(
                                direct_expires,
                                fallback_token=direct_at,
                            )
                            await self.db.update_token(
                                token_id,
                                at=direct_at,
                                at_expires=direct_at_expires,
                                credits=verify_direct.get("credits", 0),
                            )
                            debug_logger.log_info(
                                f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路 AT 验证成功（余额: {verify_direct.get('credits', 0)}）"
                            )
                            return True, ""
                        verify_direct_error = str(verify_direct.get("error") or "").strip()
                        failure_detail = (
                            "reAuth已恢复出候选AT，但同链路验活失败，判定当前记录的cookie已失效或已跳转到Google登录页，"
                            "需要重新自动登录"
                        )
                        debug_logger.log_warning(
                            f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} {failure_detail} detail={verify_direct_error}"
                        )
                        return False, failure_detail
                    else:
                        debug_logger.log_warning(
                            f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路 ST->AT 失败: {st_to_at_direct.get('error')}"
                        )
                except Exception as direct_err:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路验证异常: {direct_err}"
                    )

                # 路径1：继续走现有 flow_client 链路验证。
                if await self._do_refresh_at(
                    token_id,
                    fallback_st,
                    st_raw=(refreshed_st_raw or None),
                ):
                    debug_logger.log_info(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} ST->AT 成功"
                    )
                    return True, ""

                debug_logger.log_warning(
                    f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} ST->AT 失败，切换下一策略"
                )

            # 路径2（兼容兜底）：极少数环境返回值可能直接是 AT，再尝试按 AT 直验
            if last_candidate_token:
                debug_logger.log_info(f"[REAUTH_AT] Token {token_id}: 所有 ST->AT 策略失败，兼容尝试候选 token 直验 AT...")
                if await self._accept_direct_at_if_valid(token_id, last_candidate_token):
                    return True, ""

            return False, failure_detail

        except Exception as e:
            debug_logger.log_error(f"[REAUTH_AT] Token {token_id}: reAuth 恢复 AT 失败 - {str(e)}")
            return False, f"reAuth恢复AT异常: {str(e)}"

    async def _try_refresh_st(self, token_id: int, token) -> Optional[str]:
        """尝试通过浏览器刷新 Session Token

        使用常驻 tab 获取新的 __Secure-next-auth.session-token

        Args:
            token_id: Token ID
            token: Token 对象

        Returns:
            新的 ST 字符串，如果失败返回 None
        """
        try:
            from ..core.config import config

            # 仅在 personal 模式下支持 ST 自动刷新
            if config.captcha_method != "personal":
                debug_logger.log_info(f"[ST_REFRESH] 非 personal 模式，跳过 ST 自动刷新")
                return None

            if not token.current_project_id:
                debug_logger.log_warning(f"[ST_REFRESH] Token {token_id} 没有 project_id，无法刷新 ST")
                return None

            debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: 尝试通过浏览器刷新 ST...")

            from .browser_captcha_personal import BrowserCaptchaService
            service = await BrowserCaptchaService.get_instance(self.db)
            debug_logger.log_info(
                f"[ST_REFRESH] Token {token_id}: BrowserCaptchaService状态 "
                f"initialized={getattr(service, '_initialized', None)} "
                f"running={getattr(service, '_running', None)} "
                f"resident_count={len(getattr(service, '_resident_tabs', {}) or {})}"
            )

            # 强制重置服务实例，避免启动常驻页/残留窗口状态影响本次 ST 刷新。
            try:
                debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: 先执行 BrowserCaptcha 实例重置")
                await service.close()
                debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 实例重置完成")
            except Exception as reset_err:
                debug_logger.log_warning(
                    f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 实例重置异常，继续刷新: {reset_err}"
                )

            # 经验修复：已有常驻标签页（尤其是启动时预热页）会干扰后续 ST 刷新。
            # 在刷新前先统一清理常驻页，等价于“手动先关闭常驻浏览器后再刷新”的成功路径。
            try:
                resident_count_before = service.get_resident_count()
            except Exception:
                resident_count_before = 0
            if resident_count_before > 0:
                debug_logger.log_info(
                    f"[ST_REFRESH] Token {token_id}: 检测到常驻标签页 {resident_count_before} 个，先关闭 BrowserCaptcha 实例再刷新"
                )
                try:
                    await service.close()
                    debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 实例关闭完成")
                except Exception as stop_err:
                    debug_logger.log_warning(
                        f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 关闭异常，继续刷新: {stop_err}"
                    )

            new_st = await service.refresh_session_token(token.current_project_id)
            debug_logger.log_info(
                f"[ST_REFRESH] Token {token_id}: refresh_session_token 返回长度={len(str(new_st or ''))}"
            )
            if new_st and new_st != token.st:
                # 更新数据库中的 ST
                await self.db.update_token(token_id, st=new_st)
                debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: ST 已自动更新")
                return new_st
            elif new_st == token.st:
                debug_logger.log_warning(f"[ST_REFRESH] Token {token_id}: 获取到的 ST 与原 ST 相同，可能登录已失效")
                return None
            else:
                debug_logger.log_warning(f"[ST_REFRESH] Token {token_id}: 无法获取新 ST")
                return None

        except Exception as e:
            debug_logger.log_error(f"[ST_REFRESH] Token {token_id}: 刷新 ST 失败 - {str(e)}")
            debug_logger.log_error(f"[ST_REFRESH] Token {token_id}: 刷新 ST 失败堆栈 - {traceback.format_exc()}")
            return None

    async def ensure_project_exists(self, token_id: int) -> str:
        """Ensure a token has a pooled set of projects and return one in round-robin order."""
        async with self._project_lock:
            token = await self.db.get_token(token_id)
            if not token:
                raise ValueError("Token not found")

            projects = [project for project in await self.db.get_projects_by_token(token_id) if project.is_active]
            projects = self._sort_projects(projects)

            try:
                while len(projects) < self._project_pool_size:
                    new_project = await self._create_project_for_token(token, len(projects) + 1)
                    projects.append(new_project)
                    projects = self._sort_projects(projects)

                selected_project = self._select_next_project(token, projects)
                await self.db.update_token(
                    token_id,
                    current_project_id=selected_project.project_id,
                    current_project_name=selected_project.project_name,
                )
                return selected_project.project_id
            except Exception as e:
                raise ValueError(f"Failed to prepare project pool: {str(e)}")

    async def record_usage(self, token_id: int, is_video: bool = False):
        """Record token usage"""
        await self.db.update_token(token_id, use_count=1, last_used_at=datetime.now())

        if is_video:
            await self.db.increment_token_stats(token_id, "video")
        else:
            await self.db.increment_token_stats(token_id, "image")

    async def record_error(self, token_id: int):
        """Record token error statistics without auto-disabling the token."""
        await self.db.increment_token_stats(token_id, "error")

        try:
            stats = await self.db.get_token_stats(token_id)
            if stats:
                debug_logger.log_warning(
                    f"[TOKEN_ERROR] Token {token_id} consecutive_error_count="
                    f"{stats.consecutive_error_count}, error_count={stats.error_count}, "
                    "仅记录统计，不因普通请求失败自动禁用"
                )
        except Exception as exc:
            debug_logger.log_warning(
                f"[TOKEN_ERROR] Token {token_id}: 读取错误统计失败，但已完成错误计数累加 - {exc}"
            )

    async def record_success(self, token_id: int):
        """Record successful request (reset consecutive error count)

        This method resets error_count to 0, which is used for auto-disable threshold checking.
        Note: today_error_count and historical statistics are NOT reset.
        """
        await self.db.reset_error_count(token_id)

    async def ban_token_for_429(self, token_id: int):
        """因429错误立即禁用token

        Args:
            token_id: Token ID
        """
        debug_logger.log_warning(f"[429_BAN] 禁用Token {token_id} (原因: 429 Rate Limit)")
        await self.db.update_token(
            token_id,
            is_active=False,
            ban_reason="429_rate_limit",
            banned_at=datetime.now(timezone.utc)
        )

    async def ban_token_for_permission_denied(self, token_id: int):
        """因403 PERMISSION_DENIED错误禁用token（账号被封禁）

        此封禁不会被自动解禁，需要管理员手动处理。

        Args:
            token_id: Token ID
        """
        debug_logger.log_warning(f"[PERMISSION_DENIED_BAN] 禁用Token {token_id} (原因: 403 PERMISSION_DENIED 账号封禁)")
        await self.db.update_token(
            token_id,
            is_active=False,
            ban_reason="permission_denied",
            banned_at=datetime.now(timezone.utc)
        )

    async def ban_token_for_daily_quota(self, token_id: int):
        """因每日配额耗尽禁用token，次日零点（UTC）自动解禁。

        Args:
            token_id: Token ID
        """
        now = datetime.now(timezone.utc)
        debug_logger.log_warning(
            f"[DAILY_QUOTA_BAN] 禁用Token {token_id} (原因: 每日配额耗尽，将于明日 UTC 零点后自动解禁)"
        )
        await self.db.update_token(
            token_id,
            is_active=False,
            ban_reason="daily_quota_reached",
            banned_at=now,
        )

    async def auto_unban_daily_quota_tokens(self):
        """自动解禁因每日配额耗尽被禁用的 token。

        规则：UTC 日期已翻到禁用日期的次日，即可解禁。
        """
        all_tokens = await self.db.get_all_tokens()
        now = datetime.now(timezone.utc)
        today_utc = now.date()

        for token in all_tokens:
            if token.ban_reason != "daily_quota_reached":
                continue
            if token.is_active:
                continue
            if not token.banned_at:
                continue

            banned_at_aware = token.banned_at if token.banned_at.tzinfo else token.banned_at.replace(tzinfo=timezone.utc)
            banned_date = banned_at_aware.date()

            # 只要当前 UTC 日期 > 禁用日期，即可解禁
            if today_utc > banned_date:
                # 检查 token 是否已过期
                if token.at_expires:
                    at_expires_aware = token.at_expires if token.at_expires.tzinfo else token.at_expires.replace(tzinfo=timezone.utc)
                    if at_expires_aware <= now:
                        debug_logger.log_info(f"[DAILY_QUOTA_UNBAN] Token {token.id} 已过期，跳过解禁")
                        continue

                debug_logger.log_info(
                    f"[DAILY_QUOTA_UNBAN] 解禁Token {token.id} "
                    f"(禁用日期: {banned_date}, 今日: {today_utc})"
                )
                await self.db.update_token(
                    token.id,
                    is_active=True,
                    ban_reason=None,
                    banned_at=None,
                )
                await self.db.reset_error_count(token.id)

    async def auto_unban_429_tokens(self):
        """自动解禁因429被禁用的token

        规则:
        - 距离禁用时间12小时后自动解禁
        - 仅解禁未过期的token
        - 仅解禁因429被禁用的token
        """
        all_tokens = await self.db.get_all_tokens()
        now = datetime.now(timezone.utc)

        for token in all_tokens:
            # 跳过非429禁用的token
            if token.ban_reason != "429_rate_limit":
                continue

            # 跳过未禁用的token
            if token.is_active:
                continue

            # 跳过没有禁用时间的token
            if not token.banned_at:
                continue

            # 检查token是否已过期
            if token.at_expires:
                # 确保时区一致
                if token.at_expires.tzinfo is None:
                    at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
                else:
                    at_expires_aware = token.at_expires

                # 如果已过期，跳过
                if at_expires_aware <= now:
                    debug_logger.log_info(f"[AUTO_UNBAN] Token {token.id} 已过期，跳过解禁")
                    continue

            # 确保banned_at时区一致
            if token.banned_at.tzinfo is None:
                banned_at_aware = token.banned_at.replace(tzinfo=timezone.utc)
            else:
                banned_at_aware = token.banned_at

            # 检查是否已过12小时
            time_since_ban = now - banned_at_aware
            if time_since_ban.total_seconds() >= 12 * 3600:  # 12小时
                debug_logger.log_info(
                    f"[AUTO_UNBAN] 解禁Token {token.id} (禁用时间: {banned_at_aware}, "
                    f"已过 {time_since_ban.total_seconds() / 3600:.1f} 小时)"
                )
                await self.db.update_token(
                    token.id,
                    is_active=True,
                    ban_reason=None,
                    banned_at=None
                )
                # 重置错误计数
                await self.db.reset_error_count(token.id)

    async def auto_refresh_active_tokens(self) -> tuple[int, int, int]:
        """后台巡检活跃Token并触发AT自动刷新。

        Returns:
            (checked_count, refresh_attempted_count, refresh_failed_count)
        """
        active_tokens = await self.db.get_active_tokens()
        now = datetime.now(timezone.utc)

        checked = 0
        refresh_attempted = 0
        refresh_failed = 0

        for token in active_tokens:
            checked += 1

            needs_refresh = False
            if not token.at or not token.at_expires:
                needs_refresh = True
            else:
                if token.at_expires.tzinfo is None:
                    at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
                else:
                    at_expires_aware = token.at_expires
                # 纯定时巡检策略：
                # - 已过期：跳过（不做自动刷新尝试）
                # - 未过期且进入提前窗口：触发自动刷新
                time_left_seconds = (at_expires_aware - now).total_seconds()
                if time_left_seconds <= 0:
                    debug_logger.log_info(
                        f"[AT_AUTO_SCAN] Token {token.id}: AT已过期，跳过纯定时自动刷新"
                    )
                    continue
                advance_seconds = self._get_auto_refresh_advance_seconds(
                    token_id=token.id,
                    at_expires=at_expires_aware,
                )
                needs_refresh = time_left_seconds < advance_seconds

            if not needs_refresh:
                continue

            if self._has_recent_auto_refresh_attempt(token, now=now):
                debug_logger.log_info(
                    f"[AT_AUTO_SCAN] Token {token.id}: 当前自动刷新窗口内已尝试过 "
                    f"{getattr(token, 'last_refresh_method', '-')}, 跳过重复自动刷新"
                )
                continue

            refresh_attempted += 1
            try:
                ok = await self.is_at_valid(token.id)
                if not ok:
                    refresh_failed += 1
            except Exception as e:
                refresh_failed += 1
                debug_logger.log_error(f"[AT_AUTO_SCAN] Token {token.id}: 自动刷新巡检异常 - {str(e)}")

        return checked, refresh_attempted, refresh_failed

    # ========== 余额刷新 ==========

    async def refresh_credits(self, token_id: int) -> int:
        """刷新Token余额

        Returns:
            credits
        """
        token = await self.db.get_token(token_id)
        if not token:
            return 0

        # 确保AT有效
        token = await self.ensure_valid_token(token)
        if not token:
            return 0

        try:
            result = await self.flow_client.get_credits(token.at)
            credits = result.get("credits", 0)
            user_paygate_tier = result.get("userPaygateTier")

            # 更新数据库
            await self.db.update_token(
                token_id,
                credits=credits,
                user_paygate_tier=user_paygate_tier,
            )

            return credits
        except Exception as e:
            debug_logger.log_error(f"Failed to refresh credits for token {token_id}: {str(e)}")
            return 0
