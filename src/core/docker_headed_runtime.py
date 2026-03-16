"""Helpers for local headed browser runtime inside Linux Docker containers."""
import atexit
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import debug_logger

_MANAGED_XVFB: Optional[subprocess.Popen] = None
_ATEXIT_REGISTERED = False


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def is_running_in_docker() -> bool:
    if os.name == "nt":
        return False
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        if any(marker in content for marker in ("docker", "kubepods", "containerd")):
            return True
    except Exception:
        pass
    return bool(os.getenv("DOCKER_CONTAINER") or os.getenv("KUBERNETES_SERVICE_HOST"))


def _display_lock_path(display: str) -> Optional[Path]:
    value = str(display or "").strip()
    if not value.startswith(":"):
        return None
    display_no = value[1:].split(".", 1)[0].strip()
    if not display_no.isdigit():
        return None
    return Path(f"/tmp/.X{display_no}-lock")


def _register_atexit_cleanup() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(stop_managed_xvfb)
    _ATEXIT_REGISTERED = True


def prepare_local_headed_runtime() -> Dict[str, Any]:
    global _MANAGED_XVFB

    result: Dict[str, Any] = {
        "is_docker": is_running_in_docker(),
        "allow_headed": _env_bool("ALLOW_DOCKER_HEADED_CAPTCHA") or _env_bool("ALLOW_DOCKER_BROWSER_CAPTCHA"),
        "display": str(os.getenv("DISPLAY", "") or "").strip(),
        "display_applied": False,
        "xvfb_started": False,
        "xvfb_already_running": False,
        "xvfb_available": False,
        "reason": "",
    }

    if os.name == "nt":
        result["reason"] = "windows"
        return result

    if not result["is_docker"]:
        result["reason"] = "not_docker"
        return result

    if not result["allow_headed"]:
        result["reason"] = "headed_not_allowed"
        return result

    display = result["display"]
    if not display and _env_bool("FLOW2API_AUTO_SET_DISPLAY", default=True):
        display = str(os.getenv("FLOW2API_DEFAULT_DISPLAY", ":99") or ":99").strip() or ":99"
        os.environ["DISPLAY"] = display
        result["display"] = display
        result["display_applied"] = True
        debug_logger.log_info(f"[HeadedRuntime] Docker 环境下自动设置 DISPLAY={display}")

    if not display:
        result["reason"] = "display_missing"
        return result

    if _MANAGED_XVFB is not None and _MANAGED_XVFB.poll() is None:
        result["xvfb_available"] = True
        result["xvfb_already_running"] = True
        result["reason"] = "xvfb_managed_running"
        return result

    if not _env_bool("FLOW2API_AUTO_START_XVFB", default=True):
        result["reason"] = "auto_start_disabled"
        return result

    lock_path = _display_lock_path(display)
    if lock_path and lock_path.exists():
        result["xvfb_available"] = True
        result["xvfb_already_running"] = True
        result["reason"] = "display_lock_exists"
        return result

    xvfb_path = shutil.which("Xvfb")
    if not xvfb_path:
        result["reason"] = "xvfb_missing"
        debug_logger.log_warning("[HeadedRuntime] DISPLAY 已设置但未找到 Xvfb，可安装 xvfb 或关闭 FLOW2API_AUTO_START_XVFB")
        return result

    xvfb_whd = str(os.getenv("XVFB_WHD", "1920x1080x24") or "1920x1080x24").strip() or "1920x1080x24"
    try:
        process = subprocess.Popen(
            [xvfb_path, display, "-screen", "0", xvfb_whd, "-ac", "+extension", "RANDR"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        if process.poll() is not None:
            raise RuntimeError(f"Xvfb exited immediately with code {process.returncode}")
        _MANAGED_XVFB = process
        _register_atexit_cleanup()
        result["xvfb_available"] = True
        result["xvfb_started"] = True
        result["reason"] = "xvfb_started"
        debug_logger.log_info(f"[HeadedRuntime] 已自动启动 Xvfb: display={display}, screen={xvfb_whd}")
    except Exception as exc:
        result["reason"] = f"xvfb_start_failed:{exc}"
        debug_logger.log_warning(f"[HeadedRuntime] 自动启动 Xvfb 失败: {exc}")

    return result


def stop_managed_xvfb() -> None:
    global _MANAGED_XVFB
    process = _MANAGED_XVFB
    if process is None:
        return
    _MANAGED_XVFB = None
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
    except Exception:
        try:
            if process.poll() is None:
                process.kill()
        except Exception:
            pass
