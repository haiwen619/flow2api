from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .repository import AccountPoolRepository
from .rpa_adapter import validate_account_via_rpa

logger = logging.getLogger(__name__)


class AccountPoolService:
    def __init__(self, repo: AccountPoolRepository) -> None:
        self.repo = repo
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.batch_tasks: Dict[str, Dict[str, Any]] = {}
        self.max_jobs = 1000

    async def initialize(self) -> None:
        await self.repo.initialize()

    async def add_account(
        self,
        *,
        platform: str,
        display_name: str,
        password: str,
        is_2fa_enabled: bool,
        twofa_password: Optional[str],
        uid: Optional[str],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return await self.repo.upsert_account(
            platform=platform,
            display_name=display_name,
            password=password,
            is_2fa_enabled=is_2fa_enabled,
            twofa_password=twofa_password,
            uid=uid,
            tags=tags,
        )

    async def list_accounts(
        self,
        *,
        offset: int,
        limit: int,
        search: Optional[str],
        platform: Optional[str],
    ) -> Dict[str, Any]:
        return await self.repo.list_accounts(offset=offset, limit=limit, search=search, platform=platform)

    async def update_account(
        self,
        *,
        account_key: str,
        platform: Optional[str],
        display_name: Optional[str],
        password: Optional[str],
        is_2fa_enabled: Optional[bool],
        twofa_password: Optional[str],
        uid: Optional[str],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return await self.repo.update_account(
            account_key=account_key,
            platform=platform,
            display_name=display_name,
            password=password,
            is_2fa_enabled=is_2fa_enabled,
            twofa_password=twofa_password,
            uid=uid,
            tags=tags,
        )

    async def delete_account(self, *, account_key: str) -> None:
        await self.repo.delete_account(account_key=account_key)

    def _default_concurrency(self) -> int:
        try:
            v = int(os.getenv("ACCOUNTPOOL_VALIDATE_CONCURRENCY_DEFAULT", "5") or "5")
        except Exception:
            v = 5
        return max(1, v)

    def _max_concurrency(self) -> int:
        try:
            v = int(os.getenv("ACCOUNTPOOL_VALIDATE_CONCURRENCY_MAX", "20") or "20")
        except Exception:
            v = 20
        return max(1, v)

    def clamp_concurrency(self, value: Optional[int]) -> int:
        if value is None:
            return min(self._default_concurrency(), self._max_concurrency())
        try:
            v = int(value)
        except Exception:
            return min(self._default_concurrency(), self._max_concurrency())
        return min(max(1, v), self._max_concurrency())

    def _prune_jobs(self) -> None:
        if len(self.jobs) <= self.max_jobs:
            return
        ordered = sorted(self.jobs.items(), key=lambda kv: kv[1].get("created_at", 0))
        to_delete = max(1, len(self.jobs) - self.max_jobs)
        for job_id, _ in ordered[:to_delete]:
            self.jobs.pop(job_id, None)

    @staticmethod
    def _parse_at_expires(expires: Optional[str]) -> Optional[datetime]:
        s = str(expires or "").strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _is_transient_network_error(err: Exception) -> bool:
        msg = str(err or "").lower()
        markers = [
            "curl: (35)",
            "curl: (28)",
            "curl: (7)",
            "ssl_error_syscall",
            "ssl connect error",
            "connection reset",
            "timed out",
            "timeout was reached",
            "could not resolve host",
            "failed to connect",
            "failed to perform",
        ]
        return any(m in msg for m in markers)

    async def _st_to_at_with_retry(self, *, tm: Any, st: str, job_id: str) -> Dict[str, Any]:
        max_attempts = 3
        base_sleep = 1.0
        last_err: Optional[Exception] = None

        for i in range(1, max_attempts + 1):
            try:
                if i > 1:
                    logger.info(
                        "[AccountPool][TokenSync] job_id=%s st_to_at retry attempt=%s/%s",
                        job_id,
                        i,
                        max_attempts,
                    )
                return await tm.flow_client.st_to_at(st)
            except Exception as e:
                last_err = e
                transient = self._is_transient_network_error(e)
                logger.warning(
                    "[AccountPool][TokenSync] job_id=%s st_to_at attempt=%s/%s failed transient=%s error=%s",
                    job_id,
                    i,
                    max_attempts,
                    transient,
                    str(e),
                )
                if (not transient) or i >= max_attempts:
                    break
                await asyncio.sleep(base_sleep * i)

        raise last_err or RuntimeError("st_to_at failed with unknown error")

    async def _sync_st_cookie_to_existing_token(
        self,
        *,
        tm: Any,
        st: str,
        cookie_header: Optional[str],
        email_hint: Optional[str],
        job_id: str,
    ) -> Dict[str, Any]:
        hint = str(email_hint or "").strip()
        cookie_value = str(cookie_header or "").strip()
        if not hint or "@" not in hint:
            return {"synced": False, "reason": "invalid_email_hint"}

        token = None
        try:
            token = await tm.db.get_token_by_email(hint)
        except Exception:
            token = None
        if token is None:
            try:
                all_tokens = await tm.get_all_tokens()
                candidates = [t for t in all_tokens if str(t.email or "").strip().lower() == hint.lower()]
                if candidates:
                    candidates.sort(key=lambda t: int(t.id or 0), reverse=True)
                    token = candidates[0]
            except Exception:
                token = None

        if token is None:
            logger.warning(
                "[AccountPool][TokenSync] job_id=%s fallback sync skipped: token not found by email=%s",
                job_id,
                hint,
            )
            return {"synced": False, "reason": "token_not_found", "email": hint}

        try:
            await tm.update_token(
                token_id=int(token.id),
                st=st,
                cookie=(cookie_value or None),
            )
            logger.info(
                "[AccountPool][TokenSync] job_id=%s fallback synced st/cookie token_id=%s email=%s",
                job_id,
                int(token.id),
                hint,
            )
            return {
                "synced": True,
                "fallback": True,
                "token_id": int(token.id),
                "email": hint,
                "cookie_synced": bool(cookie_value),
            }
        except Exception as e:
            logger.warning(
                "[AccountPool][TokenSync] job_id=%s fallback update failed token_id=%s email=%s error=%s",
                job_id,
                int(token.id),
                hint,
                str(e),
            )
            return {
                "synced": False,
                "reason": "fallback_update_failed",
                "token_id": int(token.id),
                "email": hint,
                "error": str(e),
            }

    async def _sync_session_token_to_token_table(
        self,
        *,
        session_token: str,
        email_hint: Optional[str],
        cookie_header: Optional[str],
        job_id: str,
    ) -> Dict[str, Any]:
        st = str(session_token or "").strip()
        hint = str(email_hint or "").strip()
        cookie_value = str(cookie_header or "").strip()
        if not st:
            return {"synced": False, "reason": "empty_session_token"}

        # Lazy import to avoid circular import at module load time.
        try:
            from .. import admin as admin_api
        except Exception as e:
            logger.warning("[AccountPool][TokenSync] job_id=%s import admin failed: %s", job_id, str(e))
            return {"synced": False, "reason": "admin_import_failed", "error": str(e)}

        tm = getattr(admin_api, "token_manager", None)
        if tm is None:
            logger.warning("[AccountPool][TokenSync] job_id=%s token_manager not initialized", job_id)
            return {"synced": False, "reason": "token_manager_not_ready"}

        # 1) Use ST to resolve AT + canonical email (same as token edit behavior).
        try:
            conv = await self._st_to_at_with_retry(tm=tm, st=st, job_id=job_id)
        except Exception as e:
            logger.warning("[AccountPool][TokenSync] job_id=%s st_to_at failed: %s", job_id, str(e))
            try:
                proxy_cfg = await tm.db.get_proxy_config()
                logger.warning(
                    "[AccountPool][TokenSync] job_id=%s proxy_config enabled=%s proxy_url=%s",
                    job_id,
                    bool(getattr(proxy_cfg, "enabled", False)),
                    str(getattr(proxy_cfg, "proxy_url", "") or ""),
                )
            except Exception:
                pass
            if self._is_transient_network_error(e):
                # 降级处理：网络抖动时先把 ST/Cookie 落库到已存在账号，避免本次验活结果丢失
                fallback = await self._sync_st_cookie_to_existing_token(
                    tm=tm,
                    st=st,
                    cookie_header=(cookie_value or None),
                    email_hint=(hint or None),
                    job_id=job_id,
                )
                fallback["reason"] = "st_to_at_failed_transient_network_fallback"
                fallback["error"] = str(e)
                return fallback
            return {"synced": False, "reason": "st_to_at_failed", "error": str(e)}

        at = str(conv.get("access_token") or "").strip()
        expires = conv.get("expires")
        user_info = conv.get("user") or {}
        email_from_st = str(user_info.get("email") or "").strip()
        email = email_from_st or (hint if ("@" in hint and len(hint) >= 5) else "")
        if not email:
            logger.warning("[AccountPool][TokenSync] job_id=%s cannot resolve email from st/hint", job_id)
            return {"synced": False, "reason": "email_not_found"}
        if not at:
            logger.warning("[AccountPool][TokenSync] job_id=%s st_to_at returned empty AT", job_id)
            return {"synced": False, "reason": "empty_access_token", "email": email}

        # 2) Find token row by email. Try exact query first, then case-insensitive fallback.
        token = None
        try:
            token = await tm.db.get_token_by_email(email)
        except Exception:
            token = None

        if token is None:
            try:
                all_tokens = await tm.get_all_tokens()
                candidates = [t for t in all_tokens if str(t.email or "").strip().lower() == email.lower()]
                if candidates:
                    active = [t for t in candidates if bool(getattr(t, "is_active", False))]
                    choose = active if active else candidates
                    choose.sort(key=lambda t: int(t.id or 0), reverse=True)
                    token = choose[0]
            except Exception:
                token = None

        if token is None:
            logger.warning(
                "[AccountPool][TokenSync] job_id=%s no token row matched email=%s, try auto add token by ST",
                job_id,
                email,
            )
            try:
                new_token = await tm.add_token(st=st, cookie=(cookie_value or None))
                logger.info(
                    "[AccountPool][TokenSync] job_id=%s auto added token_id=%s email=%s",
                    job_id,
                    int(new_token.id or 0),
                    str(new_token.email or ""),
                )
                return {
                    "synced": True,
                    "created": True,
                    "token_id": int(new_token.id or 0),
                    "email": str(new_token.email or email),
                    "credits": getattr(new_token, "credits", None),
                }
            except Exception as e:
                # 兜底：可能并发下 ST 已被其他流程添加，尝试按 ST 回查
                try:
                    by_st = await tm.db.get_token_by_st(st)
                except Exception:
                    by_st = None
                if by_st is not None:
                    logger.info(
                        "[AccountPool][TokenSync] job_id=%s token already exists by ST token_id=%s email=%s",
                        job_id,
                        int(by_st.id or 0),
                        str(by_st.email or ""),
                    )
                    return {
                        "synced": True,
                        "created": False,
                        "matched_by": "st",
                        "token_id": int(by_st.id or 0),
                        "email": str(by_st.email or email),
                    }
                logger.warning(
                    "[AccountPool][TokenSync] job_id=%s auto add token failed email=%s error=%s",
                    job_id,
                    email,
                    str(e),
                )
                return {
                    "synced": False,
                    "reason": "add_token_failed",
                    "email": email,
                    "error": str(e),
                }

        # 3) Update Token table (ST/AT/AT_EXPIRES).
        at_expires = self._parse_at_expires(expires if isinstance(expires, str) else None)
        try:
            await tm.update_token(
                token_id=int(token.id),
                st=st,
                cookie=(cookie_value or None),
                at=at,
                at_expires=at_expires,
            )
        except Exception as e:
            logger.warning(
                "[AccountPool][TokenSync] job_id=%s update token failed token_id=%s email=%s error=%s",
                job_id,
                int(token.id),
                email,
                str(e),
            )
            return {
                "synced": False,
                "reason": "update_token_failed",
                "email": email,
                "token_id": int(token.id),
                "error": str(e),
            }

        # 4) Refresh credits best-effort (do not block sync result).
        credits = None
        try:
            credits = await tm.refresh_credits(int(token.id))
        except Exception:
            credits = None

        logger.info(
            "[AccountPool][TokenSync] job_id=%s synced token_id=%s email=%s st_len=%s",
            job_id,
            int(token.id),
            email,
            len(st),
        )
        return {
            "synced": True,
            "token_id": int(token.id),
            "email": email,
            "credits": credits,
            "cookie_synced": bool(cookie_value),
        }

    async def _sync_cookie_to_token_table(
        self,
        *,
        cookie_header: str,
        email_hint: Optional[str],
        job_id: str,
    ) -> Dict[str, Any]:
        cookie_value = str(cookie_header or "").strip()
        hint = str(email_hint or "").strip()
        if not cookie_value:
            return {"synced": False, "reason": "empty_cookie"}
        if not hint or "@" not in hint:
            return {"synced": False, "reason": "invalid_email_hint"}

        try:
            from .. import admin as admin_api
        except Exception as e:
            logger.warning("[AccountPool][CookieSync] job_id=%s import admin failed: %s", job_id, str(e))
            return {"synced": False, "reason": "admin_import_failed", "error": str(e)}

        tm = getattr(admin_api, "token_manager", None)
        if tm is None:
            logger.warning("[AccountPool][CookieSync] job_id=%s token_manager not initialized", job_id)
            return {"synced": False, "reason": "token_manager_not_ready"}

        email = hint.strip()
        token = None
        try:
            token = await tm.db.get_token_by_email(email)
        except Exception:
            token = None
        if token is None:
            try:
                all_tokens = await tm.get_all_tokens()
                candidates = [t for t in all_tokens if str(t.email or "").strip().lower() == email.lower()]
                if candidates:
                    candidates.sort(key=lambda t: int(t.id or 0), reverse=True)
                    token = candidates[0]
            except Exception:
                token = None

        if token is None:
            logger.warning(
                "[AccountPool][CookieSync] job_id=%s token not found by email=%s",
                job_id,
                email,
            )
            return {"synced": False, "reason": "token_not_found", "email": email}

        try:
            await tm.update_token(token_id=int(token.id), cookie=cookie_value)
        except Exception as e:
            logger.warning(
                "[AccountPool][CookieSync] job_id=%s update cookie failed token_id=%s email=%s error=%s",
                job_id,
                int(token.id),
                email,
                str(e),
            )
            return {
                "synced": False,
                "reason": "update_token_failed",
                "token_id": int(token.id),
                "email": email,
                "error": str(e),
            }

        logger.info(
            "[AccountPool][CookieSync] job_id=%s synced cookie token_id=%s email=%s",
            job_id,
            int(token.id),
            email,
        )
        return {
            "synced": True,
            "token_id": int(token.id),
            "email": email,
            "cookie_synced": True,
        }

    async def trigger_single_validate(
        self,
        *,
        account_key: str,
        params: Dict[str, Any],
    ) -> str:
        secret = await self.repo.get_account_secret(account_key=account_key)
        display_name = (secret.get("display_name") or "").strip()
        uid = (secret.get("uid") or "").strip()
        password = secret.get("password")
        username = uid if ("@" in uid and len(uid) >= 5) else display_name

        if not username:
            raise ValueError("account missing username")
        if password is None or not str(password).strip():
            raise ValueError("account missing password")
        validate_params = dict(params or {})
        validate_params["is_2fa_enabled"] = bool(secret.get("is_2fa_enabled"))
        if secret.get("twofa_password"):
            validate_params["twofa_password"] = secret.get("twofa_password")

        job_id = uuid.uuid4().hex
        logger.info(
            "[AccountPool] enqueue single validate account_key=%s job_id=%s username=%s is_2fa_enabled=%s",
            account_key,
            job_id,
            username,
            bool(secret.get("is_2fa_enabled")),
        )
        self.jobs[job_id] = {
            "job_id": job_id,
            "account_key": account_key,
            "username": username,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "error": None,
            "result": None,
            "cancelled": False,
            "batch_task_id": None,
        }
        self._prune_jobs()
        asyncio.create_task(
            self._run_single_job(
                job_id=job_id,
                account_key=account_key,
                username=username,
                password=str(password),
                params=validate_params,
                batch_task_id=None,
            )
        )
        return job_id

    async def _run_single_job(
        self,
        *,
        job_id: str,
        account_key: str,
        username: str,
        password: str,
        params: Dict[str, Any],
        batch_task_id: Optional[str],
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            logger.warning("[AccountPool] job missing before run job_id=%s account_key=%s", job_id, account_key)
            return
        if job.get("cancelled"):
            logger.info("[AccountPool] job cancelled before start job_id=%s account_key=%s", job_id, account_key)
            job["status"] = "cancelled"
            job["updated_at"] = time.time()
            return

        started = time.time()
        job["status"] = "running"
        job["updated_at"] = time.time()
        logger.info(
            "[AccountPool] job running job_id=%s account_key=%s username=%s",
            job_id,
            account_key,
            username,
        )
        await self.repo.set_validation(
            account_key=account_key,
            status="running",
            ok=None,
            error=None,
            job_id=job_id,
            message=None,
            set_validate_at=False,
        )

        try:
            result = await validate_account_via_rpa(
                username=username,
                password=password,
                job_id=job_id,
                params=params,
            )
            ok = bool(result.get("success"))
            session_token = (str(result.get("session_token") or "").strip() if isinstance(result, dict) else "")
            cookie_header = (str(result.get("cookie") or "").strip() if isinstance(result, dict) else "")
            payload_email = (str(result.get("payload_email") or "").strip() if isinstance(result, dict) else "")
            if ok and session_token:
                await self.repo.set_session_token(account_key=account_key, session_token=session_token)
                logger.info(
                    "[AccountPool] session token updated account_key=%s job_id=%s length=%s",
                    account_key,
                    job_id,
                    len(session_token),
                )
                sync_res = await self._sync_session_token_to_token_table(
                    session_token=session_token,
                    email_hint=(payload_email or username),
                    cookie_header=(cookie_header or None),
                    job_id=job_id,
                )
                if isinstance(result, dict):
                    result["token_sync"] = sync_res
            elif ok:
                logger.warning(
                    "[AccountPool] validate success but no session token account_key=%s job_id=%s cookie_present=%s",
                    account_key,
                    job_id,
                    bool(cookie_header),
                )
                if cookie_header:
                    cookie_sync_res = await self._sync_cookie_to_token_table(
                        cookie_header=cookie_header,
                        email_hint=(payload_email or username),
                        job_id=job_id,
                    )
                    if isinstance(result, dict):
                        result["token_sync"] = cookie_sync_res
            duration_ms = int((time.time() - started) * 1000)
            job["result"] = result
            job["status"] = "success" if ok else "failed"
            job["error"] = None if ok else (result.get("error") or "unknown error")
            job["updated_at"] = time.time()
            logger.info(
                "[AccountPool] job done job_id=%s account_key=%s status=%s duration_ms=%s message=%s error=%s",
                job_id,
                account_key,
                job["status"],
                duration_ms,
                (result.get("message") if isinstance(result, dict) else None),
                (None if ok else job["error"]),
            )

            await self.repo.set_validation(
                account_key=account_key,
                status="success" if ok else "failed",
                ok=ok,
                error=(None if ok else job["error"]),
                job_id=job_id,
                message=(result.get("message") if isinstance(result, dict) else None),
                set_validate_at=True,
            )
            self._mark_batch_progress(batch_task_id=batch_task_id, ok=ok)
        except Exception as e:
            duration_ms = int((time.time() - started) * 1000)
            job["status"] = "failed"
            job["error"] = str(e)
            job["updated_at"] = time.time()
            logger.exception(
                "[AccountPool] job exception job_id=%s account_key=%s duration_ms=%s error=%s",
                job_id,
                account_key,
                duration_ms,
                str(e),
            )
            await self.repo.set_validation(
                account_key=account_key,
                status="failed",
                ok=False,
                error=str(e),
                job_id=job_id,
                message=None,
                set_validate_at=True,
            )
            self._mark_batch_progress(batch_task_id=batch_task_id, ok=False)

    def get_job_safe(self, *, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError("job not found")
        safe = {
            "job_id": job.get("job_id"),
            "account_key": job.get("account_key"),
            "username": job.get("username"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "error": job.get("error"),
        }
        if job.get("status") in ("success", "failed"):
            result = job.get("result")
            if isinstance(result, dict):
                safe["result"] = {
                    "success": bool(result.get("success")),
                    "file_path": result.get("file_path"),
                    "auto_detected_project": result.get("auto_detected_project"),
                    "message": result.get("message"),
                    "token_sync": result.get("token_sync"),
                }
        return safe

    async def create_batch_task(
        self,
        *,
        account_keys: List[str],
        concurrency: Optional[int],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        keys = [str(x).strip() for x in account_keys if str(x).strip()]
        if not keys:
            raise ValueError("account_keys is empty")

        batch_id = uuid.uuid4().hex
        self.batch_tasks[batch_id] = {
            "batch_task_id": batch_id,
            "account_keys": keys,
            "job_ids": [],
            "total": len(keys),
            "completed": 0,
            "success": 0,
            "failed": 0,
            "status": "pending",
            "error": None,
            "cancelled": False,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        jobs: List[Dict[str, str]] = []
        for key in keys:
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "job_id": job_id,
                "account_key": key,
                "username": "",
                "status": "queued",
                "created_at": time.time(),
                "updated_at": time.time(),
                "error": None,
                "result": None,
                "cancelled": False,
                "batch_task_id": batch_id,
            }
            self.batch_tasks[batch_id]["job_ids"].append(job_id)
            jobs.append({"account_key": key, "job_id": job_id})

        eff_conc = self.clamp_concurrency(concurrency)
        asyncio.create_task(self._run_batch_task(batch_task_id=batch_id, jobs=jobs, concurrency=eff_conc, params=params))
        return {
            "batch_task_id": batch_id,
            "total": len(keys),
            "concurrency": eff_conc,
        }

    async def _run_batch_task(
        self,
        *,
        batch_task_id: str,
        jobs: List[Dict[str, str]],
        concurrency: int,
        params: Dict[str, Any],
    ) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            return
        task["status"] = "running"
        task["updated_at"] = time.time()
        sem = asyncio.Semaphore(concurrency)

        async def run_one(job_info: Dict[str, str], start_delay_sec: float) -> None:
            await asyncio.sleep(max(0.0, float(start_delay_sec)))
            t = self.batch_tasks.get(batch_task_id)
            if not t or t.get("cancelled"):
                return
            job_id = job_info["job_id"]
            account_key = job_info["account_key"]
            async with sem:
                t2 = self.batch_tasks.get(batch_task_id)
                if not t2 or t2.get("cancelled"):
                    return
                try:
                    secret = await self.repo.get_account_secret(account_key=account_key)
                    display_name = (secret.get("display_name") or "").strip()
                    uid = (secret.get("uid") or "").strip()
                    password = secret.get("password")
                    is_2fa_enabled = bool(secret.get("is_2fa_enabled"))
                    twofa_password = secret.get("twofa_password")
                    username = uid if ("@" in uid and len(uid) >= 5) else display_name
                    if not username or not password:
                        raise ValueError("account missing username/password")
                    run_params = dict(params or {})
                    run_params["is_2fa_enabled"] = is_2fa_enabled
                    if twofa_password:
                        run_params["twofa_password"] = twofa_password
                    if job_id in self.jobs:
                        self.jobs[job_id]["username"] = username
                    await self._run_single_job(
                        job_id=job_id,
                        account_key=account_key,
                        username=username,
                        password=str(password),
                        params=run_params,
                        batch_task_id=batch_task_id,
                    )
                except Exception as e:
                    if job_id in self.jobs:
                        self.jobs[job_id]["status"] = "failed"
                        self.jobs[job_id]["error"] = str(e)
                        self.jobs[job_id]["updated_at"] = time.time()
                    self._mark_batch_progress(batch_task_id=batch_task_id, ok=False)

        tasks = [
            asyncio.create_task(run_one(job, idx * 5.0))
            for idx, job in enumerate(jobs)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        final_task = self.batch_tasks.get(batch_task_id)
        if not final_task:
            return
        if final_task.get("cancelled"):
            final_task["status"] = "cancelled"
        elif final_task.get("status") not in ("failed",):
            final_task["status"] = "completed"
        final_task["updated_at"] = time.time()

    def _mark_batch_progress(self, *, batch_task_id: Optional[str], ok: bool) -> None:
        if not batch_task_id:
            return
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            return
        task["completed"] = int(task.get("completed", 0)) + 1
        if ok:
            task["success"] = int(task.get("success", 0)) + 1
        else:
            task["failed"] = int(task.get("failed", 0)) + 1
        task["updated_at"] = time.time()

    def list_batch_tasks(self) -> List[Dict[str, Any]]:
        rows = []
        for _, task in self.batch_tasks.items():
            rows.append(
                {
                    "batch_task_id": task.get("batch_task_id"),
                    "total": task.get("total"),
                    "completed": task.get("completed"),
                    "success": task.get("success"),
                    "failed": task.get("failed"),
                    "status": task.get("status"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at"),
                    "error": task.get("error"),
                    "cancelled": task.get("cancelled"),
                }
            )
        rows.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return rows

    def get_batch_task(self, *, batch_task_id: str) -> Dict[str, Any]:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        return {
            "batch_task_id": task.get("batch_task_id"),
            "total": task.get("total"),
            "completed": task.get("completed"),
            "success": task.get("success"),
            "failed": task.get("failed"),
            "status": task.get("status"),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "error": task.get("error"),
            "cancelled": task.get("cancelled"),
        }

    def delete_batch_task(self, *, batch_task_id: str) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        del self.batch_tasks[batch_task_id]

    def cancel_batch_task(self, *, batch_task_id: str) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        status = str(task.get("status") or "")
        if status in ("completed", "failed", "cancelled"):
            raise ValueError("task already finished")
        task["cancelled"] = True
        task["status"] = "cancelling"
        task["updated_at"] = time.time()
        for job_id in list(task.get("job_ids", [])):
            if job_id in self.jobs:
                self.jobs[job_id]["cancelled"] = True
                self.jobs[job_id]["status"] = "cancelled"
                self.jobs[job_id]["error"] = "task cancelled"
                self.jobs[job_id]["updated_at"] = time.time()

    def list_batch_jobs(self, *, batch_task_id: str) -> List[Dict[str, Any]]:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        rows: List[Dict[str, Any]] = []
        for job_id in task.get("job_ids", []):
            job = self.jobs.get(job_id)
            if not job:
                continue
            rows.append(
                {
                    "job_id": job.get("job_id"),
                    "account_key": job.get("account_key"),
                    "username": job.get("username"),
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "updated_at": job.get("updated_at"),
                    "error": job.get("error"),
                }
            )
        return rows

    def delete_batch_job(self, *, batch_task_id: str, job_id: str) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        if job_id not in task.get("job_ids", []):
            raise KeyError("job not found in batch task")
        task["job_ids"] = [x for x in task.get("job_ids", []) if x != job_id]
        task["total"] = max(0, int(task.get("total", 0)) - 1)
        task["updated_at"] = time.time()
        self.jobs.pop(job_id, None)
