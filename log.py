"""Compatibility shim for legacy `from log import log` imports.

Load `src/core/log.py` directly to avoid importing `src.core.__init__`,
which may pull unrelated runtime dependencies.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType


def _load_core_log_module() -> ModuleType:
    root = pathlib.Path(__file__).resolve().parent
    target = root / "src" / "core" / "log.py"
    spec = importlib.util.spec_from_file_location("_flow2api_core_log", str(target))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load log module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_flow2api_core_log", module)
    spec.loader.exec_module(module)
    return module


_m = _load_core_log_module()

log = _m.log
set_log_level = _m.set_log_level
LOG_LEVELS = _m.LOG_LEVELS

__all__ = ["log", "set_log_level", "LOG_LEVELS"]
