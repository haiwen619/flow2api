from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _run_async_in_dedicated_loop(coro_factory) -> Any:
    """
    Run coroutine in a fresh event loop.

    On Windows, prefer Proactor loop because Playwright needs subprocess support.
    """
    loop = None
    try:
        if os.name == "nt":
            proactor_loop_cls = getattr(asyncio, "ProactorEventLoop", None)
            if proactor_loop_cls is not None:
                loop = proactor_loop_cls()
        if loop is None:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro_factory())
    finally:
        if loop is not None:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass


async def validate_account_via_rpa(
    *,
    username: str,
    password: str,
    job_id: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    RPA adapter.

    Mode:
    - ACCOUNTPOOL_RPA_MODE=browser_automation (default): call Rpa.BrowserAutomation.main
    - ACCOUNTPOOL_RPA_MODE=stub: return mock success (only for local integration test)
    """
    if not username or not password:
        logger.warning(
            "[AccountPool][RPA] job_id=%s missing username/password username_present=%s password_present=%s",
            job_id,
            bool(username),
            bool(password),
        )
        return {"success": False, "error": "missing username/password"}

    mode = (os.getenv("ACCOUNTPOOL_RPA_MODE", "browser_automation") or "browser_automation").strip().lower()
    logger.info(
        "[AccountPool][RPA] job_id=%s start mode=%s username=%s is_2fa_enabled=%s",
        job_id,
        mode,
        username,
        bool(params.get("is_2fa_enabled")),
    )

    if mode == "stub":
        logger.warning("[AccountPool][RPA] job_id=%s running in STUB mode", job_id)
        return {
            "success": True,
            "message": "rpa_stub_success",
            "error": None,
            "file_path": None,
            "auto_detected_project": None,
            "session_token": None,
        }
    if mode != "browser_automation":
        logger.error("[AccountPool][RPA] job_id=%s unsupported mode=%s", job_id, mode)
        return {
            "success": False,
            "message": "rpa_unknown_mode",
            "error": f"unsupported mode: {mode}",
            "file_path": None,
            "auto_detected_project": None,
        }

    try:
        from Rpa.BrowserAutomation.main import (
            DEFAULT_PANEL_VALIDATE_OPTIONS,
            ValidateOptions,
            validate_antigravity_account,
        )

        opts = ValidateOptions(
            headless=bool(params.get("headless", DEFAULT_PANEL_VALIDATE_OPTIONS.headless)),
            manual=bool(params.get("manual", DEFAULT_PANEL_VALIDATE_OPTIONS.manual)),
            timeout_sec=int(params.get("timeout_sec", DEFAULT_PANEL_VALIDATE_OPTIONS.timeout_sec)),
            slow_mo_ms=int(params.get("slow_mo_ms", DEFAULT_PANEL_VALIDATE_OPTIONS.slow_mo_ms)),
            user_agent=(params.get("ua") or DEFAULT_PANEL_VALIDATE_OPTIONS.user_agent),
            locale=(params.get("locale") or DEFAULT_PANEL_VALIDATE_OPTIONS.locale),
            timezone_id=(params.get("timezone") or DEFAULT_PANEL_VALIDATE_OPTIONS.timezone_id),
            viewport_width=int(params.get("viewport_width", DEFAULT_PANEL_VALIDATE_OPTIONS.viewport_width or 1366)),
            viewport_height=int(params.get("viewport_height", DEFAULT_PANEL_VALIDATE_OPTIONS.viewport_height or 768)),
            external_browser=bool(params.get("external_browser", DEFAULT_PANEL_VALIDATE_OPTIONS.external_browser)),
            human_delay_min_sec=float(params.get("human_delay_min", DEFAULT_PANEL_VALIDATE_OPTIONS.human_delay_min_sec)),
            human_delay_max_sec=float(params.get("human_delay_max", DEFAULT_PANEL_VALIDATE_OPTIONS.human_delay_max_sec)),
            bitbrowser=bool(params.get("bitbrowser", DEFAULT_PANEL_VALIDATE_OPTIONS.bitbrowser)),
            bitbrowser_id=(params.get("bitbrowser_id") or DEFAULT_PANEL_VALIDATE_OPTIONS.bitbrowser_id),
            bitbrowser_auto_delete=bool(
                params.get("bitbrowser_auto_delete", DEFAULT_PANEL_VALIDATE_OPTIONS.bitbrowser_auto_delete)
            ),
        )

        validate_kwargs = {
            "username": username,
            "password": password,
            "user_session": f"accountpool-{job_id}",
            "is_2fa_enabled": bool(params.get("is_2fa_enabled")),
            "twofa_password": (str(params.get("twofa_password") or "").strip() or None),
            "options": opts,
        }

        started = time.time()
        if os.name == "nt":
            logger.info(
                "[AccountPool][RPA] job_id=%s run validate in worker thread with dedicated loop (windows)",
                job_id,
            )
            result = await asyncio.to_thread(
                _run_async_in_dedicated_loop,
                lambda: validate_antigravity_account(**validate_kwargs),
            )
        else:
            result = await validate_antigravity_account(**validate_kwargs)

        duration_ms = int((time.time() - started) * 1000)
        if not isinstance(result, dict):
            logger.error("[AccountPool][RPA] job_id=%s invalid result type=%s", job_id, type(result).__name__)
            return {"success": False, "error": "rpa returned invalid result"}
        logger.info(
            "[AccountPool][RPA] job_id=%s finished success=%s duration_ms=%s",
            job_id,
            bool(result.get("success")),
            duration_ms,
        )
        return {
            "success": bool(result.get("success")),
            "message": result.get("message"),
            "error": result.get("error"),
            "file_path": result.get("file_path"),
            "auto_detected_project": result.get("auto_detected_project"),
            "session_token": (str(result.get("session_token") or "").strip() or None),
        }
    except Exception as e:
        logger.exception("[AccountPool][RPA] job_id=%s browser_automation failed: %s", job_id, str(e))
        return {
            "success": False,
            "message": "rpa_browser_automation_failed",
            "error": str(e),
            "file_path": None,
            "auto_detected_project": None,
        }
