"""Debug logger optimized for bursty concurrent request logging."""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from .config import config

LOG_LEVELS = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}


def _monotonic_ts() -> float:
    import time

    return time.monotonic()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


class _AsyncFileSink:
    """Low-overhead async sink using deque + Condition on the hot path."""

    def __init__(
        self,
        *,
        log_file: str,
        queue_size: int,
        batch_size: int,
        flush_interval: float,
        max_bytes: int,
        backup_count: int,
        daily_rotation: bool,
        echo_to_console: bool,
    ) -> None:
        self._log_file = log_file
        self._queue_size = max(1, int(queue_size))
        self._batch_size = max(1, int(batch_size))
        self._flush_interval = max(0.1, float(flush_interval))
        self._max_bytes = max(1024, int(max_bytes))
        self._backup_count = max(1, int(backup_count))
        self._daily_rotation = bool(daily_rotation)
        self._echo_to_console = bool(echo_to_console)

        self._log_deque: Deque[str] = deque()
        self._deque_condition = threading.Condition(threading.Lock())
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_running = False
        self._start_lock = threading.Lock()
        self._drop_count_lock = threading.Lock()

        self._file_writing_disabled = False
        self._disable_reason: Optional[str] = None
        self._main_handle = None
        self._daily_handle = None
        self._active_daily_file: Optional[str] = None
        self._current_size = 0
        self._dropped_count = 0

    def _get_dropped_count(self) -> int:
        with self._drop_count_lock:
            return self._dropped_count

    def _ensure_parent_dir(self, raw_path: str) -> None:
        path = Path(raw_path)
        if path.parent and str(path.parent) not in {"", "."}:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _daily_log_file(self, now: Optional[datetime] = None) -> str:
        path = Path(self._log_file)
        ts = now or datetime.now()
        suffix = path.suffix or ".log"
        return str(path.with_name(f"{path.stem}-{ts.strftime('%Y-%m-%d')}{suffix}"))

    def _close_main_handle(self) -> None:
        if self._main_handle is not None:
            try:
                self._main_handle.flush()
                self._main_handle.close()
            except Exception:
                pass
            finally:
                self._main_handle = None

    def _close_daily_handle(self) -> None:
        if self._daily_handle is not None:
            try:
                self._daily_handle.flush()
                self._daily_handle.close()
            except Exception:
                pass
            finally:
                self._daily_handle = None
                self._active_daily_file = None

    def _open_main_handle(self, mode: str = "a") -> bool:
        self._close_main_handle()
        try:
            self._ensure_parent_dir(self._log_file)
            self._main_handle = open(self._log_file, mode, encoding="utf-8", buffering=65536)
            if mode == "a":
                try:
                    self._current_size = os.path.getsize(self._log_file)
                except OSError:
                    self._current_size = 0
            else:
                self._current_size = 0
            return True
        except (PermissionError, OSError, IOError) as exc:
            self._file_writing_disabled = True
            self._disable_reason = str(exc)
            print(f"Warning: Cannot open log file, disabling file writing: {exc}", file=sys.stderr)
            return False

    def _open_daily_handle(self, mode: str = "a", *, now: Optional[datetime] = None) -> bool:
        if not self._daily_rotation:
            return True
        daily_file = self._daily_log_file(now)
        if self._active_daily_file == daily_file and self._daily_handle is not None:
            return True
        self._close_daily_handle()
        try:
            self._ensure_parent_dir(daily_file)
            self._daily_handle = open(daily_file, mode, encoding="utf-8", buffering=65536)
            self._active_daily_file = daily_file
            return True
        except (PermissionError, OSError, IOError) as exc:
            self._file_writing_disabled = True
            self._disable_reason = str(exc)
            print(f"Warning: Cannot open daily log file, disabling file writing: {exc}", file=sys.stderr)
            return False

    def _rotate_main_file(self) -> None:
        self._close_main_handle()

        base = self._log_file
        dot_pos = base.rfind(".")
        if dot_pos > 0:
            stem, ext = base[:dot_pos], base[dot_pos:]
        else:
            stem, ext = base, ""

        try:
            for index in range(self._backup_count, 0, -1):
                src = f"{stem}.{index}{ext}"
                if index >= self._backup_count:
                    if os.path.exists(src):
                        os.remove(src)
                else:
                    dst = f"{stem}.{index + 1}{ext}"
                    if os.path.exists(src):
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.rename(src, dst)

            first_backup = f"{stem}.1{ext}"
            if os.path.exists(base):
                if os.path.exists(first_backup):
                    os.remove(first_backup)
                os.rename(base, first_backup)
        except Exception as exc:
            print(f"Warning: Log rotation failed: {exc}", file=sys.stderr)

        self._current_size = 0
        self._open_main_handle("a")

    def _emit_console(self, entry: str, level: str) -> None:
        if not self._echo_to_console:
            return
        if level in {"error", "critical"}:
            print(entry, file=sys.stderr)
        else:
            print(entry)

    def _start_writer_thread(self) -> None:
        with self._start_lock:
            if self._writer_thread is not None and self._writer_thread.is_alive():
                return
            self._writer_running = True
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
                name="DebugLogWriter",
            )
            self._writer_thread.start()

    def _ensure_started(self) -> None:
        if self._file_writing_disabled:
            return
        if self._main_handle is None:
            self._open_main_handle("a")
        if self._daily_rotation and self._daily_handle is None:
            self._open_daily_handle("a")
        self._start_writer_thread()

    def emit(self, level: str, entry: str) -> None:
        self._emit_console(entry, level)
        if self._file_writing_disabled:
            return
        self._ensure_started()
        if self._file_writing_disabled:
            return

        if len(self._log_deque) >= self._queue_size:
            with self._drop_count_lock:
                self._dropped_count += 1
            return

        self._log_deque.append(entry)
        if self._deque_condition.acquire(blocking=False):
            try:
                self._deque_condition.notify()
            finally:
                self._deque_condition.release()

    def _drain_batch(self) -> list[str]:
        batch: list[str] = []
        with self._drop_count_lock:
            dropped = self._dropped_count
            self._dropped_count = 0
        if dropped > 0:
            notice_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch.append(
                f"[{notice_ts}] [WARNING] debug log queue overflow, dropped {dropped} entries"
            )
        for _ in range(self._batch_size):
            if not self._log_deque:
                break
            batch.append(self._log_deque.popleft())
        return batch

    def _writer_loop(self) -> None:
        last_flush_at = 0.0

        while True:
            with self._deque_condition:
                if not self._log_deque and self._writer_running and self._get_dropped_count() <= 0:
                    self._deque_condition.wait(timeout=self._flush_interval)
                batch = self._drain_batch()

            if batch and not self._file_writing_disabled:
                chunk = "\n".join(batch) + "\n"
                chunk_bytes = len(chunk.encode("utf-8", errors="replace"))
                try:
                    if self._main_handle is None:
                        self._open_main_handle("a")
                    if self._daily_rotation:
                        self._open_daily_handle("a")
                    if self._main_handle is not None:
                        self._main_handle.write(chunk)
                        self._current_size += chunk_bytes
                    if self._daily_rotation and self._daily_handle is not None:
                        self._daily_handle.write(chunk)
                except Exception as exc:
                    print(f"Warning: Failed to write debug log batch: {exc}", file=sys.stderr)
                    self._close_main_handle()
                    self._close_daily_handle()
                    try:
                        self._open_main_handle("a")
                        if self._daily_rotation:
                            self._open_daily_handle("a")
                    except Exception:
                        pass

                if self._current_size >= self._max_bytes:
                    try:
                        if self._main_handle is not None:
                            self._main_handle.flush()
                        self._rotate_main_file()
                    except Exception as exc:
                        print(f"Warning: Debug log rotation failed: {exc}", file=sys.stderr)

            now = _monotonic_ts()
            if now - last_flush_at >= self._flush_interval:
                if self._main_handle is not None:
                    try:
                        self._main_handle.flush()
                    except Exception:
                        pass
                if self._daily_rotation and self._daily_handle is not None:
                    try:
                        self._daily_handle.flush()
                    except Exception:
                        pass
                last_flush_at = now

            if not self._writer_running and not self._log_deque and self._get_dropped_count() <= 0:
                break

        self._close_main_handle()
        self._close_daily_handle()

    def close(self) -> None:
        self._writer_running = False
        with self._deque_condition:
            self._deque_condition.notify_all()
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3.0)

    def get_queue_size(self) -> int:
        return len(self._log_deque)

    def get_log_file(self) -> str:
        return self._log_file


class DebugLogger:
    """Project debug logger with async file sink and existing API surface."""

    def __init__(self) -> None:
        self.log_file = Path(os.getenv("LOG_FILE", "RunLogs/app.log"))
        self._sink = _AsyncFileSink(
            log_file=str(self.log_file),
            queue_size=_env_int("DEBUG_LOG_QUEUE_SIZE", 5000, 100),
            batch_size=_env_int("DEBUG_LOG_BATCH_SIZE", 500, 1),
            flush_interval=_env_float("DEBUG_LOG_FLUSH_INTERVAL", 1.0, 0.1),
            max_bytes=_env_int("LOG_MAX_SIZE_MB", 50, 1) * 1024 * 1024,
            backup_count=_env_int("LOG_MAX_BACKUPS", 3, 1),
            daily_rotation=_env_flag("LOG_DAILY_ROTATION", False),
            echo_to_console=_env_flag("DEBUG_LOG_ECHO_CONSOLE", True),
        )
        self._level_name = (os.getenv("LOG_LEVEL", "info") or "info").lower()
        self._level_value = LOG_LEVELS.get(self._level_name, LOG_LEVELS["info"])

    def _mask_token(self, token: str) -> str:
        if not config.debug_mask_token or len(token) <= 12:
            return token
        return f"{token[:6]}...{token[-6:]}"

    def _format_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _should_log(self, level: str) -> bool:
        level_value = LOG_LEVELS.get((level or "").lower(), LOG_LEVELS["info"])
        return config.debug_enabled and level_value >= self._level_value

    def _emit(self, level: str, message: str) -> None:
        if not self._should_log(level):
            return
        timestamp = self._format_timestamp()
        entry = f"[{timestamp}] [{level.upper()}] {message}"
        self._sink.emit(level.lower(), entry)

    def _emit_block(self, level: str, lines: list[str]) -> None:
        block = "\n".join(str(line) for line in lines if line is not None)
        self._emit(level, block)

    def _truncate_large_fields(self, data: Any, max_length: int = 200) -> Any:
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if (
                    key in {"encodedImage", "base64", "imageData", "data"}
                    and isinstance(value, str)
                    and len(value) > max_length
                ):
                    result[key] = f"{value[:100]}... (truncated, total {len(value)} chars)"
                else:
                    result[key] = self._truncate_large_fields(value, max_length)
            return result
        if isinstance(data, list):
            return [self._truncate_large_fields(item, max_length) for item in data]
        if isinstance(data, str) and len(data) > 10000:
            return f"{data[:100]}... (truncated, total {len(data)} chars)"
        return data

    def _format_payload(self, payload: Any) -> str:
        if isinstance(payload, (dict, list)):
            safe_payload = self._truncate_large_fields(payload)
            return json.dumps(safe_payload, indent=2, ensure_ascii=False)
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except Exception:
                return payload if len(payload) <= 2000 else f"{payload[:2000]}... (truncated)"
            safe_payload = self._truncate_large_fields(parsed)
            return json.dumps(safe_payload, indent=2, ensure_ascii=False)
        return str(payload)

    def log_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[Any] = None,
        files: Optional[Dict] = None,
        proxy: Optional[str] = None,
    ) -> None:
        if not (config.debug_enabled and config.debug_log_requests):
            return

        try:
            masked_headers = dict(headers or {})
            auth_key = None
            if "Authorization" in masked_headers:
                auth_key = "Authorization"
            elif "authorization" in masked_headers:
                auth_key = "authorization"

            if auth_key:
                auth_value = str(masked_headers.get(auth_key) or "")
                if auth_value.startswith("Bearer "):
                    masked_headers[auth_key] = f"Bearer {self._mask_token(auth_value[7:])}"

            if "Cookie" in masked_headers:
                cookie_value = str(masked_headers["Cookie"] or "")
                if "__Secure-next-auth.session-token=" in cookie_value:
                    parts = cookie_value.split("=", 1)
                    if len(parts) == 2:
                        st_token = parts[1].split(";")[0]
                        masked_headers["Cookie"] = (
                            "__Secure-next-auth.session-token="
                            + self._mask_token(st_token)
                        )

            lines = [
                "=" * 100,
                f"[REQUEST] {self._format_timestamp()}",
                "-" * 100,
                f"Method: {method}",
                f"URL: {url}",
                "",
                "Headers:",
            ]
            lines.extend([f"  {key}: {value}" for key, value in masked_headers.items()])

            if body is not None:
                lines.extend(["", "Request Body:", self._format_payload(body)])

            if files:
                lines.append("")
                lines.append("Files:")
                try:
                    if hasattr(files, "keys") and callable(getattr(files, "keys", None)):
                        lines.extend([f"  {key}: <file data>" for key in files.keys()])
                    else:
                        lines.append("  <multipart form data>")
                except Exception:
                    lines.append("  <binary file data>")

            if proxy:
                lines.extend(["", f"Proxy: {proxy}"])

            lines.extend(["=" * 100, ""])
            self._emit_block("info", lines)
        except Exception as exc:
            self._emit("error", f"Error logging request: {exc}")

    def log_response(
        self,
        status_code: int,
        headers: Dict[str, str],
        body: Any,
        duration_ms: Optional[float] = None,
    ) -> None:
        if not (config.debug_enabled and config.debug_log_responses):
            return

        try:
            lines = [
                "=" * 100,
                f"[RESPONSE] {self._format_timestamp()}",
                "-" * 100,
                f"Status: {status_code}",
            ]

            if duration_ms is not None:
                lines.append(f"Duration: {duration_ms:.2f}ms")

            lines.extend(["", "Response Headers:"])
            lines.extend([f"  {key}: {value}" for key, value in (headers or {}).items()])
            lines.extend(["", "Response Body:", self._format_payload(body), "=" * 100, ""])
            self._emit_block("info", lines)
        except Exception as exc:
            self._emit("error", f"Error logging response: {exc}")

    def log_error(
        self,
        error_message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ) -> None:
        if not config.debug_enabled:
            return

        if status_code is None and response_text is None:
            self._emit("error", str(error_message))
            return

        try:
            lines = [
                "=" * 100,
                f"[ERROR] {self._format_timestamp()}",
                "-" * 100,
            ]
            if status_code is not None:
                lines.append(f"Status Code: {status_code}")
            lines.append(f"Error Message: {error_message}")
            if response_text:
                lines.extend(["", "Error Response:", self._format_payload(response_text)])
            lines.extend(["=" * 100, ""])
            self._emit_block("error", lines)
        except Exception as exc:
            self._emit("error", f"Error logging error: {exc}")

    def log_debug(self, message: str) -> None:
        self._emit("debug", message)

    def debug(self, message: str) -> None:
        self.log_debug(message)

    def log_info(self, message: str) -> None:
        self._emit("info", message)

    def info(self, message: str) -> None:
        self.log_info(message)

    def log_warning(self, message: str) -> None:
        self._emit("warning", message)

    def warning(self, message: str) -> None:
        self.log_warning(message)

    def error(self, message: str) -> None:
        self.log_error(message)

    def log_exception(self, message: str) -> None:
        details = traceback.format_exc()
        if details and details.strip() != "NoneType: None":
            self._emit("error", f"{message}\n{details.rstrip()}")
        else:
            self._emit("error", message)

    def get_log_file(self) -> str:
        return self._sink.get_log_file()

    def get_queue_size(self) -> int:
        return self._sink.get_queue_size()

    def close(self) -> None:
        self._sink.close()


debug_logger = DebugLogger()

atexit.register(debug_logger.close)
