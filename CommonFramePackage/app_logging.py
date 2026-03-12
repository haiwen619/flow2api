"""Reusable async logger with console + file output."""

from __future__ import annotations

import atexit
import os
import sys
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque

LOG_LEVELS = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}


def _now_ts() -> float:
    import time

    return time.monotonic()


class AsyncLogger:
    """Async file logger optimized for low-overhead hot paths."""

    def __init__(
        self,
        *,
        log_level: str = "info",
        log_file: str = "RunLogs/app.log",
        enabled: bool = True,
        clear_on_init: bool = True,
        echo_to_console: bool = True,
        queue_size: int = 5000,
        batch_size: int = 1000,
        flush_interval: float = 2.0,
        daily_rotation: bool = True,
        register_atexit: bool = False,
    ) -> None:
        self._cached_log_level = LOG_LEVELS.get((log_level or "info").lower(), LOG_LEVELS["info"])
        self._cached_log_file = log_file or "log.txt"
        self._log_enabled = bool(enabled)
        self._clear_on_init = bool(clear_on_init)
        self._echo_to_console = bool(echo_to_console)
        self._max_queue_size = max(1, int(queue_size))
        self._batch_size = max(1, int(batch_size))
        self._flush_interval = max(0.1, float(flush_interval))
        self._daily_rotation = bool(daily_rotation)

        self._file_writing_disabled = False
        self._disable_reason: str | None = None
        self._log_file_handle = None
        self._daily_log_file_handle = None
        self._active_daily_log_file: str | None = None

        self._log_deque: Deque[str] = deque()
        self._deque_condition = threading.Condition(threading.Lock())
        self._writer_thread: threading.Thread | None = None
        self._writer_running = False

        if self._log_enabled:
            if self._clear_on_init:
                self._clear_log_file()
            self._start_writer_thread()

        if register_atexit:
            atexit.register(self.close)

    def _close_log_file(self) -> None:
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.flush()
                self._log_file_handle.close()
            except Exception:
                pass
            finally:
                self._log_file_handle = None

    def _close_daily_log_file(self) -> None:
        if self._daily_log_file_handle is not None:
            try:
                self._daily_log_file_handle.flush()
                self._daily_log_file_handle.close()
            except Exception:
                pass
            finally:
                self._daily_log_file_handle = None
                self._active_daily_log_file = None

    def _ensure_parent_dir(self, raw_path: str) -> None:
        log_path = Path(raw_path)
        if log_path.parent and str(log_path.parent) not in ("", "."):
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_daily_log_file_path(self, now: datetime | None = None) -> str:
        current_path = Path(self._cached_log_file)
        ts = now or datetime.now()
        suffix = current_path.suffix or ".log"
        dated_name = f"{current_path.stem}-{ts.strftime('%Y-%m-%d')}{suffix}"
        return str(current_path.with_name(dated_name))

    def _open_log_file(self, mode: str = "a") -> bool:
        self._close_log_file()
        try:
            self._ensure_parent_dir(self._cached_log_file)
            self._log_file_handle = open(self._cached_log_file, mode, encoding="utf-8", buffering=65536)
            return True
        except (PermissionError, OSError, IOError) as exc:
            self._file_writing_disabled = True
            self._disable_reason = str(exc)
            print(f"Warning: Cannot open log file, disabling file writing: {exc}", file=sys.stderr)
            print("Log messages will continue to display in console only.", file=sys.stderr)
            return False
        except Exception as exc:
            print(f"Warning: Failed to open log file: {exc}", file=sys.stderr)
            return False

    def _open_daily_log_file(self, mode: str = "a", *, now: datetime | None = None) -> bool:
        if not self._daily_rotation:
            return True
        daily_log_file = self._build_daily_log_file_path(now)
        if self._active_daily_log_file == daily_log_file and self._daily_log_file_handle is not None:
            return True
        self._close_daily_log_file()
        try:
            self._ensure_parent_dir(daily_log_file)
            self._daily_log_file_handle = open(daily_log_file, mode, encoding="utf-8", buffering=65536)
            self._active_daily_log_file = daily_log_file
            return True
        except (PermissionError, OSError, IOError) as exc:
            self._file_writing_disabled = True
            self._disable_reason = str(exc)
            print(f"Warning: Cannot open daily log file, disabling file writing: {exc}", file=sys.stderr)
            print("Log messages will continue to display in console only.", file=sys.stderr)
            return False
        except Exception as exc:
            print(f"Warning: Failed to open daily log file: {exc}", file=sys.stderr)
            return False

    def _clear_log_file(self) -> None:
        try:
            self._ensure_parent_dir(self._cached_log_file)
            with open(self._cached_log_file, "w", encoding="utf-8"):
                pass
            self._open_log_file("a")
            if self._daily_rotation:
                self._open_daily_log_file("a")
        except (PermissionError, OSError, IOError) as exc:
            self._file_writing_disabled = True
            self._disable_reason = str(exc)
            print(
                f"Warning: File system appears to be read-only or permission denied. "
                f"Disabling log file writing: {exc}",
                file=sys.stderr,
            )
            print("Log messages will continue to display in console only.", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: Failed to clear log file: {exc}", file=sys.stderr)

    def _log_writer_worker(self) -> None:
        last_flush_time = 0.0

        while True:
            with self._deque_condition:
                if not self._log_deque and self._writer_running:
                    self._deque_condition.wait(timeout=self._flush_interval)

                batch: list[str] = []
                for _ in range(self._batch_size):
                    if self._log_deque:
                        batch.append(self._log_deque.popleft())
                    else:
                        break

            if batch and not self._file_writing_disabled:
                chunk = "\n".join(batch) + "\n"
                try:
                    if self._log_file_handle is None:
                        self._open_log_file("a")
                    if self._daily_rotation:
                        self._open_daily_log_file("a")
                    if self._log_file_handle is not None:
                        self._log_file_handle.write(chunk)
                    if self._daily_rotation and self._daily_log_file_handle is not None:
                        self._daily_log_file_handle.write(chunk)
                except Exception as exc:
                    print(f"Warning: Failed to write log batch: {exc}", file=sys.stderr)
                    self._close_log_file()
                    self._close_daily_log_file()
                    try:
                        self._open_log_file("a")
                        if self._daily_rotation:
                            self._open_daily_log_file("a")
                    except Exception:
                        pass

            now = _now_ts()
            if now - last_flush_time >= self._flush_interval:
                if self._log_file_handle is not None:
                    try:
                        self._log_file_handle.flush()
                    except Exception:
                        pass
                if self._daily_rotation and self._daily_log_file_handle is not None:
                    try:
                        self._daily_log_file_handle.flush()
                    except Exception:
                        pass
                last_flush_time = now

            if not self._writer_running and not self._log_deque:
                break

        if self._log_file_handle is not None:
            try:
                self._log_file_handle.flush()
            except Exception:
                pass
        if self._daily_rotation and self._daily_log_file_handle is not None:
            try:
                self._daily_log_file_handle.flush()
            except Exception:
                pass
        self._close_log_file()
        self._close_daily_log_file()

    def _start_writer_thread(self) -> None:
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._writer_running = True
            self._writer_thread = threading.Thread(
                target=self._log_writer_worker,
                daemon=True,
                name="LogWriter",
            )
            self._writer_thread.start()

    def _write_to_file(self, message: str) -> None:
        if self._file_writing_disabled or not self._log_enabled:
            return
        if len(self._log_deque) >= self._max_queue_size:
            return
        self._log_deque.append(message)
        if self._deque_condition.acquire(blocking=False):
            try:
                self._deque_condition.notify()
            finally:
                self._deque_condition.release()

    def _emit(self, level: str, message: str) -> None:
        if not self._log_enabled:
            return

        normalized_level = (level or "").lower()
        level_val = LOG_LEVELS.get(normalized_level)
        if level_val is None:
            print(f"Warning: Unknown log level '{level}'", file=sys.stderr)
            return
        if level_val < self._cached_log_level:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{normalized_level.upper()}] {message}"

        if self._echo_to_console:
            if normalized_level in ("error", "critical"):
                print(entry, file=sys.stderr)
            else:
                print(entry)

        self._write_to_file(entry)

    def __call__(self, level: str, message: str) -> None:
        self._emit(level, message)

    def debug(self, message: str) -> None:
        self._emit("debug", message)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warning(self, message: str) -> None:
        self._emit("warning", message)

    def error(self, message: str) -> None:
        self._emit("error", message)

    def critical(self, message: str) -> None:
        self._emit("critical", message)

    def exception(self, message: str) -> None:
        details = traceback.format_exc()
        if details and details.strip() != "NoneType: None":
            self._emit("error", f"{message}\n{details.rstrip()}")
            return
        self._emit("error", str(message))

    def set_level(self, level: str) -> bool:
        normalized_level = (level or "").lower()
        if normalized_level not in LOG_LEVELS:
            print(
                f"Warning: Unknown log level '{level}'. "
                f"Valid levels: {', '.join(LOG_LEVELS.keys())}"
            )
            return False
        self._cached_log_level = LOG_LEVELS[normalized_level]
        return True

    def get_current_level(self) -> str:
        for name, value in LOG_LEVELS.items():
            if value == self._cached_log_level:
                return name
        return "info"

    def get_log_file(self) -> str:
        return self._cached_log_file

    def get_daily_log_file(self) -> str | None:
        if not self._daily_rotation:
            return None
        return self._build_daily_log_file_path()

    def get_queue_size(self) -> int:
        return len(self._log_deque)

    def close(self) -> None:
        self._writer_running = False
        with self._deque_condition:
            self._deque_condition.notify_all()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3.0)


def build_logger_from_env(
    *,
    level_env: str = "LOG_LEVEL",
    file_env: str = "LOG_FILE",
    enabled_env: str = "ENABLE_LOG",
    daily_rotation_env: str = "LOG_DAILY_ROTATION",
    default_level: str = "info",
    default_file: str = "RunLogs/app.log",
    clear_on_init: bool = True,
    echo_to_console: bool = True,
    queue_size: int = 5000,
    batch_size: int = 1000,
    flush_interval: float = 2.0,
    default_daily_rotation: bool = True,
    register_atexit: bool = False,
) -> AsyncLogger:
    """Build a logger instance from environment variables."""

    level = (os.getenv(level_env, default_level) or default_level).lower()
    log_file = os.getenv(file_env, default_file) or default_file
    enabled = (os.getenv(enabled_env, "1") or "1").strip().lower() not in ("0", "false", "no", "off")
    daily_rotation = (
        os.getenv(daily_rotation_env, "1" if default_daily_rotation else "0") or "1"
    ).strip().lower() not in ("0", "false", "no", "off")

    return AsyncLogger(
        log_level=level,
        log_file=log_file,
        enabled=enabled,
        clear_on_init=clear_on_init,
        echo_to_console=echo_to_console,
        queue_size=queue_size,
        batch_size=batch_size,
        flush_interval=flush_interval,
        daily_rotation=daily_rotation,
        register_atexit=register_atexit,
    )


__all__ = ["AsyncLogger", "LOG_LEVELS", "build_logger_from_env"]
