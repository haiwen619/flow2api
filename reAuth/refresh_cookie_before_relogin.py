"""
Refresh session cookies before re-login.

This is a Python rewrite of `src/javalogin/demo.java` reAuth flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

try:
    from curl_cffi.requests import Session
    _SESSION_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - import-time env issue
    Session = None  # type: ignore[assignment]
    _SESSION_IMPORT_ERROR = e

try:
    from src.core.logger import debug_logger as _project_debug_logger
except Exception:
    _project_debug_logger = None


def _format_log_message(message: str, *args: Any) -> str:
    try:
        return message % args if args else str(message)
    except Exception:
        return f"{message} {' '.join(str(a) for a in args)}".strip()


class _MirroredLogger:
    """Mirror all local logs to project debug_logger.log_warning."""

    def __init__(self, base_logger: logging.Logger):
        self._base = base_logger

    def _mirror_warning(self, level: str, message: str, *args: Any) -> None:
        if not _project_debug_logger:
            return
        try:
            text = _format_log_message(message, *args)
            _project_debug_logger.log_warning(f"[reAuth][{level}] {text}")
        except Exception:
            pass

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._base.info(message, *args, **kwargs)
        self._mirror_warning("INFO", message, *args)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._base.warning(message, *args, **kwargs)
        self._mirror_warning("WARNING", message, *args)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._base.error(message, *args, **kwargs)
        self._mirror_warning("ERROR", message, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


LOGGER = _MirroredLogger(logging.getLogger("reAuth"))

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)

BASE_URL = "https://labs.google/fx"
FLOW_API_BASE_URL = "https://aisandbox-pa.googleapis.com/v1"
DEFAULT_BROWSER_TEST_URL = "https://labs.google/fx/tools/flow"


def _log_warning(message: str, *args) -> None:
    """兼容旧调用入口，统一走镜像 logger。"""
    LOGGER.warning(message, *args)


@dataclass
class ReAuthAccount:
    project_id: str
    cookie: str
    cookie_file: Optional[str] = None
    observed_tokens: List[Dict[str, str]] = field(default_factory=list)
    final_session_token: str = ""
    final_session_set_cookie_raw: str = ""


def _clip_long_text(value: Optional[str], max_len: int = 220) -> str:
    """裁剪过长日志文本，避免刷屏。"""
    text = value or ""
    if len(text) <= max_len:
        return text
    head = max_len // 2
    tail = max_len - head
    omitted = len(text) - max_len
    return f"{text[:head]} ...[已省略 {omitted} 字符]... {text[-tail:]}"


def _cookie_header_to_dict(cookie_header: str) -> Dict[str, str]:
    jar = SimpleCookie()
    jar.load(cookie_header or "")
    return {k: morsel.value for k, morsel in jar.items()}


def _cookie_dict_to_header(cookie_dict: Dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in cookie_dict.items())


def _clean_cookie_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def _extract_cookie_value_by_key(cookie_like_text: str, key: str) -> Optional[str]:
    pattern = rf"(?:^|[;,]\s*){re.escape(key)}=([^;,\r\n]+)"
    match = re.search(pattern, cookie_like_text or "")
    if not match:
        return None
    return _clean_cookie_token(match.group(1))


def _extract_cookie_chunked_value(cookie_like_text: str, key: str) -> Optional[str]:
    """
    Extract and join chunked cookie value:
      key.0=...; key.1=...; key.2=...
    """
    pattern = rf"(?:^|[;,]\s*){re.escape(key)}\.(\d+)=([^;,\r\n]+)"
    matches = re.findall(pattern, cookie_like_text or "")
    if not matches:
        return None

    chunks: List[tuple[int, str]] = []
    for idx_text, value_text in matches:
        try:
            idx = int(idx_text)
        except Exception:
            continue
        v = _clean_cookie_token(value_text) or ""
        chunks.append((idx, v))
    if not chunks:
        return None

    chunks.sort(key=lambda x: x[0])
    return "".join(v for _, v in chunks) or None


def _verify_at_via_get_credits(
    candidate_token: str,
    *,
    proxy_url: Optional[str],
    connect_timeout: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    通过 /v1/credits 测试候选 token 是否可作为 AT 使用。
    返回结构:
    {
      "success": bool,
      "credits": int | None,
      "userPaygateTier": str | None,
      "error": str | None
    }
    """
    if not candidate_token:
        return {"success": False, "credits": None, "userPaygateTier": None, "error": "empty token"}

    if Session is None:
        return {
            "success": False,
            "credits": None,
            "userPaygateTier": None,
            "error": f"curl_cffi not available: {_SESSION_IMPORT_ERROR}",
        }

    url = f"{FLOW_API_BASE_URL}/credits"
    headers = {
        "authorization": f"Bearer {candidate_token}",
        "accept": "application/json",
        "content-type": "application/json",
    }
    try:
        with Session() as session:
            resp = session.request(
                method="GET",
                url=url,
                headers=headers,
                timeout=(connect_timeout, timeout),
                proxy=proxy_url,
                impersonate="chrome110",
            )

        if resp.status_code >= 400:
            return {
                "success": False,
                "credits": None,
                "userPaygateTier": None,
                "error": f"HTTP {resp.status_code}: {_clip_long_text(resp.text, 160)}",
            }

        try:
            data = resp.json()
        except Exception:
            data = {}
        return {
            "success": True,
            "credits": data.get("credits"),
            "userPaygateTier": data.get("userPaygateTier"),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "credits": None,
            "userPaygateTier": None,
            "error": str(e),
        }


def _st_to_at_via_auth_session(
    st_token: str,
    *,
    proxy_url: Optional[str],
    connect_timeout: int,
    timeout: int,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    """使用 ST 调用 /api/auth/session 转换为 AT。"""
    if not st_token:
        return {"success": False, "access_token": None, "expires": None, "user": None, "error": "empty st token"}

    if Session is None:
        return {
            "success": False,
            "access_token": None,
            "expires": None,
            "user": None,
            "error": f"curl_cffi not available: {_SESSION_IMPORT_ERROR}",
        }

    url = f"{BASE_URL}/api/auth/session"
    headers = {
        "User-Agent": user_agent,
        "accept": "application/json",
        "content-type": "application/json",
        "Cookie": f"__Secure-next-auth.session-token={st_token}",
    }
    try:
        with Session() as session:
            resp = session.request(
                method="GET",
                url=url,
                headers=headers,
                timeout=(connect_timeout, timeout),
                proxy=proxy_url,
                impersonate="chrome110",
            )
        if resp.status_code >= 400:
            return {
                "success": False,
                "access_token": None,
                "expires": None,
                "user": None,
                "error": f"HTTP {resp.status_code}: {_clip_long_text(resp.text, 200)}",
            }

        try:
            payload = resp.json()
        except Exception:
            payload = {}

        access_token = payload.get("access_token")
        expires = payload.get("expires")
        user = payload.get("user")
        if not access_token:
            return {
                "success": False,
                "access_token": None,
                "expires": expires,
                "user": user,
                "error": f"Missing access_token in /api/auth/session response: {_clip_long_text(resp.text, 200)}",
            }

        return {
            "success": True,
            "access_token": access_token,
            "expires": expires,
            "user": user,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "access_token": None,
            "expires": None,
            "user": None,
            "error": str(e),
        }


def _log_flow_overview(
    project_id: str,
    timeout: int,
    connect_timeout: int,
    max_retries: int,
    proxy_url: Optional[str],
    cookie_header: str,
    strict_java_flow: bool,
) -> None:
    LOGGER.info("========== reAuth 功能说明（开始） ==========")
    LOGGER.info("用途: 在重新登录前，通过 6 步会话链路尝试刷新 Cookie / Session Token。")
    LOGGER.info("目标项目: %s", project_id)
    LOGGER.info("请求策略: connect_timeout=%ss, read_timeout=%ss, retries=%s", connect_timeout, timeout, max_retries)
    LOGGER.info("代理配置: %s", proxy_url or "未启用")
    LOGGER.info("流程模式: %s", "strict-java-flow(严格1:1)" if strict_java_flow else "default(增强)")
    LOGGER.info("初始Cookie全文: %s", cookie_header or "<空Cookie>")
    LOGGER.info("流程概览:")
    LOGGER.info("  1) providers: 探测会话上下文")
    LOGGER.info("  2) csrf: 获取 csrfToken")
    LOGGER.info("  3) signin/google: 触发登录链路并更新 cookie")
    LOGGER.info("  4) follow url: 获取中转跳转地址")
    LOGGER.info("  5) follow location: 继续跳转并更新 cookie")
    LOGGER.info("  6) project page: 访问项目页完成会话固化")
    LOGGER.info("==========================================")


def _log_step_intro(
    step_no: int,
    title: str,
    purpose: str,
    request_brief: str,
    expected_output: str,
    failure_impact: str,
) -> None:
    LOGGER.info("------------ Step %s: %s ------------", step_no, title)
    LOGGER.info("功能说明: %s", purpose)
    LOGGER.info("请求要点: %s", request_brief)
    LOGGER.info("期望输出: %s", expected_output)
    LOGGER.info("失败影响: %s", failure_impact)


def verify_input_cookie_only(
    cookie_header: str,
    *,
    proxy_url: Optional[str],
    connect_timeout: int,
    timeout: int,
) -> Dict[str, Any]:
    """仅做验证处理：直接用传入 cookie 提取 ST，再转 AT 后验证。"""
    LOGGER.info("========== Cookie 直验模式（verify-only） ==========")
    LOGGER.info("模式说明: 跳过 reAuth 6 步，仅验证传入 cookie 中 ST 转换得到的 AT 是否可用。")
    LOGGER.info("传入Cookie全文: %s", cookie_header or "<空Cookie>")

    session_token = extract_session_token_from_cookie_header(cookie_header) or ""
    if not session_token:
        err = "No __Secure-next-auth.session-token found in cookie"
        _log_warning("verify-only: %s", err)
        return {
            "mode": "verify-only",
            "cookie": cookie_header,
            "token": "",
            "session_token": "",
            "observed_tokens": [
                {"stage": "verify-only:input-cookie", "source": "__Secure-next-auth.session-token", "token": ""},
            ],
            "at_test": {
                "success": False,
                "credits": None,
                "userPaygateTier": None,
                "error": err,
            },
        }

    LOGGER.info("session_token(完整): %s", session_token)
    st_to_at_result = _st_to_at_via_auth_session(
        session_token,
        proxy_url=proxy_url,
        connect_timeout=connect_timeout,
        timeout=timeout,
    )
    if not st_to_at_result.get("success"):
        err = st_to_at_result.get("error") or "st_to_at failed"
        _log_warning("verify-only: ST->AT 失败: %s", err)
        return {
            "mode": "verify-only",
            "cookie": cookie_header,
            "token": "",
            "session_token": session_token,
            "st_to_at": st_to_at_result,
            "observed_tokens": [
                {"stage": "verify-only:input-cookie", "source": "__Secure-next-auth.session-token", "token": session_token},
                {"stage": "verify-only:st_to_at.access_token", "source": "access_token", "token": ""},
            ],
            "at_test": {
                "success": False,
                "credits": None,
                "userPaygateTier": None,
                "error": f"ST->AT failed: {err}",
            },
        }

    candidate_token = st_to_at_result.get("access_token") or ""
    LOGGER.info("ST->AT 成功: expires=%s", st_to_at_result.get("expires"))
    LOGGER.info("AT(完整): %s", candidate_token)
    verify_result = _verify_at_via_get_credits(
        candidate_token,
        proxy_url=proxy_url,
        connect_timeout=connect_timeout,
        timeout=timeout,
    )
    if verify_result.get("success"):
        LOGGER.info(
            "verify-only 完成: AT 验证成功, credits=%s, tier=%s",
            verify_result.get("credits"),
            verify_result.get("userPaygateTier"),
        )
    else:
        _log_warning("verify-only 失败: %s", verify_result.get("error"))

    return {
        "mode": "verify-only",
        "cookie": cookie_header,
        "token": candidate_token,
        "session_token": session_token,
        "st_to_at": st_to_at_result,
        "observed_tokens": [
            {"stage": "verify-only:input-cookie", "source": "__Secure-next-auth.session-token", "token": session_token},
            {"stage": "verify-only:st_to_at.access_token", "source": "access_token", "token": candidate_token},
            {"stage": "verify-only:candidate", "source": "access_token", "token": candidate_token},
        ],
        "at_test": verify_result,
    }


def extract_session_token_from_cookie_header(cookie_header: str) -> Optional[str]:
    """Extract __Secure-next-auth.session-token from Cookie header string."""
    text = cookie_header or ""
    candidates: List[str] = []

    # 1) chunked cookies (session-token.0 / .1 / ...) - usually the most complete in reAuth flow.
    chunked = _extract_cookie_chunked_value(text, "__Secure-next-auth.session-token")
    if chunked:
        candidates.append(chunked)

    # 2) plain key=value in raw header / set-cookie text.
    direct = _extract_cookie_value_by_key(text, "__Secure-next-auth.session-token")
    if direct:
        candidates.append(direct)

    # 3) SimpleCookie parser fallback.
    cookie_dict = _cookie_header_to_dict(text)
    token = cookie_dict.get("__Secure-next-auth.session-token")
    if token:
        cleaned = _clean_cookie_token(token)
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return None
    # Prefer longest candidate to avoid truncated parser variants.
    return max(candidates, key=len)


def normalize_proxy_url(proxy_value: Optional[str]) -> Optional[str]:
    """Normalize proxy input to a full URL string."""
    if not proxy_value:
        return None

    value = str(proxy_value).strip()
    if not value:
        return None

    # Port only, e.g. "6987"
    if value.isdigit():
        return f"http://127.0.0.1:{value}"

    # Host:port without scheme
    if "://" not in value and ":" in value:
        return f"http://{value}"

    return value


def _cookie_header_to_pairs(cookie_header: str) -> List[tuple[str, str]]:
    """Parse cookie header text into (name, value) pairs."""
    text = cookie_header or ""
    parsed = _cookie_header_to_dict(text)
    if parsed:
        return [(k, v) for k, v in parsed.items() if str(k).strip()]

    # Fallback for loose/malformed cookie strings.
    pairs: List[tuple[str, str]] = []
    for part in text.split(";"):
        segment = (part or "").strip()
        if not segment or "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        cleaned = _clean_cookie_token(value)
        if not key or cleaned is None:
            continue
        pairs.append((key, cleaned))
    return pairs


def _build_playwright_cookies(cookie_header: str, target_url: str) -> List[Dict[str, Any]]:
    """Convert Cookie header into Playwright `context.add_cookies` payload."""
    parsed_url = urlparse((target_url or "").strip())
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError(f"Invalid browser target url: {target_url}")

    cookie_scope_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    ignored_set_cookie_attrs = {
        "path",
        "domain",
        "expires",
        "max-age",
        "secure",
        "httponly",
        "samesite",
        "priority",
        "partitioned",
    }
    cookies_for_browser: List[Dict[str, Any]] = []
    for name, value in _cookie_header_to_pairs(cookie_header):
        if name.strip().lower() in ignored_set_cookie_attrs:
            continue
        cookie_item: Dict[str, Any] = {"name": name, "value": value, "url": cookie_scope_url}
        if name.startswith("__Secure-") or name.startswith("__Host-"):
            cookie_item["secure"] = True
        # NOTE: For Playwright add_cookies, `url + path` on __Host-* may be rejected.
        # Keep url-only form here; path "/" is implied by the scoped URL.
        cookies_for_browser.append(cookie_item)
    return cookies_for_browser


def open_browser_with_cookie_for_validation(
    cookie_header: str,
    *,
    target_url: str = DEFAULT_BROWSER_TEST_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    proxy_url: Optional[str] = None,
    headless: bool = False,
    wait_seconds: int = 45,
) -> Dict[str, Any]:
    """
    Use Playwright to open target_url with the provided cookie for manual validation.

    Returns:
        {
          "success": bool,
          "final_url": str | None,
          "title": str | None,
          "likely_logged_in": bool | None,
          "cookie_count": int,
          "error": str | None
        }
    """
    if not cookie_header:
        return {
            "success": False,
            "final_url": None,
            "title": None,
            "likely_logged_in": None,
            "cookie_count": 0,
            "error": "empty cookie",
        }

    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return {
            "success": False,
            "final_url": None,
            "title": None,
            "likely_logged_in": None,
            "cookie_count": 0,
            "error": f"playwright unavailable: {e}",
        }

    browser_cookies = _build_playwright_cookies(cookie_header, target_url)
    if not browser_cookies:
        return {
            "success": False,
            "final_url": None,
            "title": None,
            "likely_logged_in": None,
            "cookie_count": 0,
            "error": "no cookies parsed from cookie header",
        }

    final_url: Optional[str] = None
    title: Optional[str] = None
    likely_logged_in: Optional[bool] = None
    wait_seconds = max(0, int(wait_seconds))
    LOGGER.info("[browser-test] 准备打开页面: %s", target_url)
    LOGGER.info("[browser-test] 注入 Cookie 数量: %s", len(browser_cookies))
    browser = None
    context = None
    with sync_playwright() as p:
        try:
            launch_kwargs: Dict[str, Any] = {"headless": bool(headless)}
            if proxy_url:
                launch_kwargs["proxy"] = {"server": proxy_url}
            browser = p.chromium.launch(**launch_kwargs)
            context_kwargs: Dict[str, Any] = {}
            if user_agent:
                context_kwargs["user_agent"] = user_agent
            context = browser.new_context(**context_kwargs)
            try:
                context.add_cookies(browser_cookies)
            except Exception as add_err:
                LOGGER.warning("[browser-test] 批量注入 Cookie 失败，改为逐条注入: %s", add_err)
                injected_count = 0
                skipped: List[str] = []
                for item in browser_cookies:
                    try:
                        context.add_cookies([item])
                        injected_count += 1
                    except Exception:
                        skipped.append(str(item.get("name") or ""))
                LOGGER.info(
                    "[browser-test] 逐条注入完成: success=%s skipped=%s",
                    injected_count,
                    skipped,
                )
            page = context.new_page()
            page.goto(target_url, wait_until="domcontentloaded", timeout=90_000)
            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            final_url = page.url
            title = page.title()
            likely_logged_in = "accounts.google.com" not in (final_url or "").lower()
            LOGGER.info("[browser-test] 页面打开完成: final_url=%s title=%s", final_url, title)
            LOGGER.info("[browser-test] 登录态判断(粗略): %s", likely_logged_in)

            if not headless and wait_seconds > 0:
                LOGGER.info("[browser-test] 浏览器将保持 %s 秒，便于人工观察页面。", wait_seconds)
                page.wait_for_timeout(wait_seconds * 1000)
            return {
                "success": True,
                "final_url": final_url,
                "title": title,
                "likely_logged_in": likely_logged_in,
                "cookie_count": len(browser_cookies),
                "error": None,
            }
        except Exception as e:
            _log_warning("[browser-test] 打开浏览器验证失败: %s", e)
            return {
                "success": False,
                "final_url": final_url,
                "title": title,
                "likely_logged_in": likely_logged_in,
                "cookie_count": len(browser_cookies),
                "error": str(e),
            }
        finally:
            if context is not None:
                try:
                    context.close()
                except Exception:
                    pass
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass


def _extract_set_cookie_values(response) -> List[str]:
    def _split_combined_set_cookie(raw_header: str) -> List[str]:
        """Split combined Set-Cookie header safely (don't split inside Expires=...)."""
        text = raw_header or ""
        if not text:
            return []

        parts: List[str] = []
        buf: List[str] = []
        in_expires = False
        i = 0
        lower = text.lower()
        while i < len(text):
            if lower.startswith("expires=", i):
                in_expires = True

            ch = text[i]
            if ch == ";" and in_expires:
                in_expires = False
            elif ch == "," and not in_expires:
                segment = "".join(buf).strip()
                if segment:
                    parts.append(segment)
                buf = []
                i += 1
                continue

            buf.append(ch)
            i += 1

        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    headers = response.headers
    values: List[str] = []
    raw_values: List[str] = []

    # 1) Try list API first (some libs may return stripped cookie pairs here).
    if hasattr(headers, "get_list"):
        try:
            raw_values.extend(headers.get_list("set-cookie") or [])
        except Exception:
            pass
        try:
            raw_values.extend(headers.get_list("Set-Cookie") or [])
        except Exception:
            pass

    # 2) Always include raw header accessor as fallback (often keeps attributes).
    raw_lower = headers.get("set-cookie")
    raw_upper = headers.get("Set-Cookie")
    for raw in (raw_lower, raw_upper):
        if not raw:
            continue
        if isinstance(raw, list):
            raw_values.extend([str(x) for x in raw if x is not None])
        else:
            raw_values.append(str(raw))

    # Split all candidates.
    for raw in raw_values:
        values.extend(_split_combined_set_cookie(raw))
    values = [v for v in values if v]
    if not values:
        return []

    # Deduplicate by cookie key and keep the longest value (usually includes attributes).
    merged: Dict[str, str] = {}
    order: List[str] = []
    for item in values:
        m = re.match(r"^\s*([^=;,\s]+)=", item)
        key = m.group(1).strip().lower() if m else item.strip().lower()
        prev = merged.get(key)
        if prev is None:
            merged[key] = item
            order.append(key)
        else:
            # Prefer fuller entry that keeps Path/Expires/HttpOnly...
            if len(item) > len(prev):
                merged[key] = item

    return [merged[k] for k in order]


def _extract_token_from_set_cookie_values(set_cookie_values: List[str]) -> Optional[str]:
    """Extract session token from raw Set-Cookie values without relying on cookie merge."""
    for set_cookie in set_cookie_values:
        token = extract_session_token_from_cookie_header(set_cookie)
        if token:
            return token
    return None


def _extract_session_set_cookie_raw(set_cookie_values: List[str]) -> Optional[str]:
    """Extract Set-Cookie raw for session token.

    - exact key exists: return longest exact entry
    - only chunk keys (.0/.1/...): return all chunk entries joined by index order
    """
    if not set_cookie_values:
        return None
    key_exact = "__secure-next-auth.session-token="
    key_chunk = "__secure-next-auth.session-token."
    exact_items: List[str] = []
    chunk_items: List[tuple[int, str]] = []
    for set_cookie in set_cookie_values:
        text = (set_cookie or "").strip()
        lower = text.lower()
        if lower.startswith(key_exact):
            exact_items.append(text)
            continue
        if lower.startswith(key_chunk):
            m = re.match(r"^\s*__Secure-next-auth\.session-token\.(\d+)=", text, re.IGNORECASE)
            if m:
                try:
                    idx = int(m.group(1))
                except Exception:
                    idx = 10**9
            else:
                idx = 10**9
            chunk_items.append((idx, text))

    if exact_items:
        return max(exact_items, key=len)
    if chunk_items:
        chunk_items.sort(key=lambda x: x[0])
        return ", ".join(item for _, item in chunk_items)
    return None


def _merge_response_cookies(response, cookie_dict: Dict[str, str]) -> None:
    for set_cookie in _extract_set_cookie_values(response):
        # Always try first cookie pair parse, even if SimpleCookie fails.
        pair_match = re.match(r"^\s*([^=;,\s]+)=([^;]+)", set_cookie)
        if pair_match:
            key = pair_match.group(1).strip()
            value = _clean_cookie_token(pair_match.group(2))
            if key and value is not None:
                cookie_dict[key] = value

        parsed = SimpleCookie()
        parsed.load(set_cookie)
        for key, morsel in parsed.items():
            cookie_dict[key] = morsel.value


def refresh_cookie_before_relogin(
    account: ReAuthAccount,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 30,
    proxy_url: Optional[str] = None,
    connect_timeout: int = 10,
    max_retries: int = 2,
    test_at_after_refresh: bool = False,
    strict_java_flow: bool = False,
) -> str:
    """
    Execute the same 6-step cookie refresh flow as Java reAuth.

    Returns:
        Updated cookie header string.
    """
    _log_flow_overview(
        project_id=account.project_id,
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        proxy_url=proxy_url,
        cookie_header=account.cookie,
        strict_java_flow=strict_java_flow,
    )
    if Session is None:
        raise RuntimeError(
            "curl_cffi is not installed in current Python environment. "
            "Please activate venv and install dependencies: "
            "`.venv\\Scripts\\activate` then `pip install -r requirements.txt`. "
            f"Original import error: {_SESSION_IMPORT_ERROR}"
        )
    cookies = _cookie_header_to_dict(account.cookie)
    observed_tokens: List[Dict[str, str]] = []

    def current_cookie() -> str:
        return _cookie_dict_to_header(cookies)

    def record_token(stage: str, source: str, token: Optional[str]) -> None:
        token_value = token or ""
        observed_tokens.append({"stage": stage, "source": source, "token": token_value})
        LOGGER.info("Token记录 [%s][%s]: %s", stage, source, token_value or "<空>")

    def record_cookie_state(stage: str, capture_tokens: bool = False) -> None:
        cookie_value = current_cookie()
        # LOGGER.info("%s Cookie全文: %s", stage, cookie_value or "<空Cookie>")
        if capture_tokens:
            session_token = extract_session_token_from_cookie_header(cookie_value)
            record_token(stage, "__Secure-next-auth.session-token", session_token)

    with Session() as session:
        def do_request(
            method: str,
            url: str,
            *,
            headers: Optional[Dict[str, str]] = None,
            data: Optional[str] = None,
            cookie_override: Optional[str] = None,
            merge_cookies: bool = True,
            record_set_cookie_tokens: bool = False,
        ):
            request_headers = dict(headers or {})
            request_headers["Cookie"] = current_cookie() if cookie_override is None else cookie_override

            last_err: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                start_ts = time.time()
                try:
                    LOGGER.info("request %s %s (attempt %s/%s)", method, url, attempt, max_retries)
                    response = session.request(
                        method=method,
                        url=url,
                        headers=request_headers,
                        data=data,
                        timeout=(connect_timeout, timeout),
                        impersonate="chrome110",
                        proxy=proxy_url,
                        allow_redirects=False,
                    )
                    set_cookie_values = _extract_set_cookie_values(response)
                    LOGGER.info("response set-cookie count: %s", len(set_cookie_values))
                    if record_set_cookie_tokens:
                        record_token(
                            f"{method} {url}",
                            "set-cookie.__Secure-next-auth.session-token",
                            _extract_token_from_set_cookie_values(set_cookie_values),
                        )
                    if merge_cookies:
                        _merge_response_cookies(response, cookies)
                    LOGGER.info(
                        "response %s %s status=%s cost=%.2fs",
                        method, url, response.status_code, time.time() - start_ts
                    )
                    return response
                except KeyboardInterrupt:
                    raise RuntimeError(
                        "Request interrupted by user (KeyboardInterrupt). "
                        "This usually means the network request was slow/stuck and Ctrl+C was pressed."
                    )
                except Exception as e:
                    last_err = e
                    _log_warning("request failed: %s %s attempt=%s err=%s", method, url, attempt, e)
                    if attempt < max_retries:
                        time.sleep(1)

            raise RuntimeError(f"Request failed after {max_retries} attempts: {method} {url}; last_error={last_err}")

        # Step 1: GET /api/auth/providers
        _log_step_intro(
            step_no=1,
            title="会话探测 providers",
            purpose="校验当前 Cookie 是否能建立基础认证上下文，初始化服务端会话状态。",
            request_brief=f"GET {BASE_URL}/api/auth/providers，携带当前 Cookie。",
            expected_output="返回 200/3xx，并可能下发新的 Set-Cookie。",
            failure_impact="后续 csrfToken 获取通常会失败，流程中断。",
        )
        step1_url = f"{BASE_URL}/api/auth/providers"
        record_cookie_state("step1 请求前")
        resp1 = do_request(
            "GET",
            step1_url,
            headers={"accept": "*/*", "content-type": "application/json"},
            merge_cookies=not strict_java_flow,
        )
        LOGGER.info("step1 完成: status=%s", resp1.status_code)
        record_cookie_state("step1 完成后", capture_tokens=False)

        # Step 2: GET /api/auth/csrf
        _log_step_intro(
            step_no=2,
            title="获取 csrfToken",
            purpose="获取登录链路必须的 csrfToken。",
            request_brief=f"GET {BASE_URL}/api/auth/csrf，携带 UA + Cookie。",
            expected_output="JSON 中包含 csrfToken 字段。",
            failure_impact="无法提交 signin/google，流程中断。",
        )
        step2_url = f"{BASE_URL}/api/auth/csrf"
        resp2 = do_request(
            "GET",
            step2_url,
            headers={
                "User-Agent": user_agent,
                "accept": "*/*",
                "content-type": "application/json",
            },
            merge_cookies=not strict_java_flow,
        )
        LOGGER.info("step2 GET %s status=%s body=%s", step2_url, resp2.status_code, _clip_long_text(resp2.text))
        try:
            csrf_token = resp2.json().get("csrfToken")
        except Exception:
            csrf_token = None
        if not csrf_token:
            raise RuntimeError(f"Missing csrfToken from {step2_url}: {_clip_long_text(resp2.text)}")
        LOGGER.info("step2 完成: csrfToken(完整)=%s", csrf_token)
        record_cookie_state("step2 完成后", capture_tokens=False)

        # Step 3: POST /api/auth/signin/google
        _log_step_intro(
            step_no=3,
            title="提交 signin/google",
            purpose="用 csrfToken + callbackUrl 触发登录链路，获取下一跳 URL，并更新 Cookie。",
            request_brief=f"POST {BASE_URL}/api/auth/signin/google，form 包含 csrfToken/callbackUrl。",
            expected_output="返回 JSON，包含 url 字段；响应头可能附带 Set-Cookie。",
            failure_impact="无法拿到中转链接，流程中断。",
        )
        step3_url = f"{BASE_URL}/api/auth/signin/google"
        form_data = urlencode(
            {
                "redirect": "false",
                "csrfToken": csrf_token,
                "callbackUrl": f"{BASE_URL}/tools/flow/project/{account.project_id}",
                "json": "true",
            }
        )
        resp3 = do_request(
            "POST",
            step3_url,
            headers={
                "User-Agent": user_agent,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "*/*",
            },
            data=form_data,
            merge_cookies=True,
        )
        LOGGER.info("step3 POST %s status=%s body=%s", step3_url, resp3.status_code, _clip_long_text(resp3.text))
        record_cookie_state("step3 第一次记录", capture_tokens=False)

        try:
            step3_json = resp3.json()
        except Exception:
            step3_json = {}
        next_url = step3_json.get("url")
        if not next_url:
            raise RuntimeError(f"Missing next url from step3 response: {_clip_long_text(resp3.text)}")
        LOGGER.info("step3 完成: next_url=%s", next_url)
        record_cookie_state("step3 完成后", capture_tokens=False)

        # Step 4: GET returned URL from step 3
        _log_step_intro(
            step_no=4,
            title="访问 step3 返回 URL",
            purpose="进入认证中转页，提取 HTML 跳转链接或 location 头。",
            request_brief="GET next_url，优先使用 cookie_file（若传入）否则使用当前 Cookie。",
            expected_output="得到 location 或 HTML 中的 A HREF。",
            failure_impact="无法继续跳转链路，流程中断。",
        )
        if strict_java_flow:
            # Java: step4 固定走 account.getCookieFile()（为空时退回初始 cookie）
            step4_cookie = account.cookie_file if account.cookie_file is not None else account.cookie
        else:
            step4_cookie = account.cookie_file or current_cookie()
        LOGGER.info("step4 使用Cookie全文: %s", step4_cookie or "<空Cookie>")
        resp4 = do_request(
            "GET",
            next_url,
            headers={"User-Agent": user_agent, "accept": "*/*"},
            cookie_override=step4_cookie,
            merge_cookies=not strict_java_flow,
        )
        html = resp4.text or ""
        match = re.search(r'<A HREF="(.*?)">', html, re.IGNORECASE)
        redirect_url = match.group(1) if match else None
        location_header = resp4.headers.get("location")
        LOGGER.info("step4 GET %s status=%s", next_url, resp4.status_code)
        LOGGER.info("step4 redirect_url=%s location=%s", redirect_url, location_header)
        record_cookie_state("step4 完成后", capture_tokens=False)

        # Step 5: GET location from step 4
        _log_step_intro(
            step_no=5,
            title="跟进 location 跳转",
            purpose="继续认证跳转链路，促使服务端下发后续会话 Cookie。",
            request_brief="GET step4 返回的 location（或 HTML 里的 redirect_url）。",
            expected_output="成功响应并可能继续提供 location，Cookie 被更新。",
            failure_impact="会话可能只刷新一半，Token 提取失败概率高。",
        )
        if strict_java_flow and not location_header and redirect_url:
            LOGGER.warning("strict-java-flow 已启用：step5 仅允许使用 location_header，忽略 redirect_url。")
        location = location_header if strict_java_flow else (location_header or redirect_url)
        if not location:
            raise RuntimeError("Missing location for step5")
        resp5 = do_request(
            "GET",
            location,
            headers={
                "User-Agent": user_agent,
                "accept": "*/*",
                "content-type": "application/json",
            },
            merge_cookies=True,
        )
        step5_location = resp5.headers.get("location")
        LOGGER.info("step5 GET %s status=%s location=%s", location, resp5.status_code, step5_location)
        record_cookie_state("step5 完成后", capture_tokens=False)

        # Step 6: GET project page
        _log_step_intro(
            step_no=6,
            title="访问项目页固化会话",
            purpose="访问目标项目页，完成登录状态落地，确保最终 Cookie 可用。",
            request_brief=f"GET {BASE_URL}/tools/flow/project/<project_id>。",
            expected_output="返回项目页响应，Cookie 进入稳定状态。",
            failure_impact="可能拿到部分 Cookie，但会话不稳定。",
        )
        project_url = f"{BASE_URL}/tools/flow/project/{account.project_id}"
        resp6 = do_request(
            "GET",
            project_url,
            headers={
                "User-Agent": user_agent,
                "accept": "*/*",
                "content-type": "application/json",
            },
            merge_cookies=True,
            record_set_cookie_tokens=True,
        )
        LOGGER.info("step6 GET %s status=%s", project_url, resp6.status_code)
        if resp6.status_code >= 400:
            LOGGER.error(
                "step6 请求失败: status=%s, body=%s",
                resp6.status_code,
                _clip_long_text(resp6.text, 300),
            )
        else:
            LOGGER.info("step6 请求成功: status=%s", resp6.status_code)

        step6_location_header = resp6.headers.get("location")
        step6_content_type = resp6.headers.get("content-type")
        step6_cache_control = resp6.headers.get("cache-control")
        raw_set_cookie_lower = resp6.headers.get("set-cookie")
        raw_set_cookie_upper = resp6.headers.get("Set-Cookie")
        LOGGER.info("step6 响应头 location=%s", step6_location_header)
        LOGGER.info("step6 响应头 content-type=%s", step6_content_type)
        LOGGER.info("step6 响应头 cache-control=%s", step6_cache_control)
        LOGGER.info("step6 原始 set-cookie(lower)=%s", _clip_long_text(raw_set_cookie_lower, 500))
        LOGGER.info("step6 原始 set-cookie(upper)=%s", _clip_long_text(raw_set_cookie_upper, 500))

        step6_set_cookie_values = _extract_set_cookie_values(resp6)
        step6_session_from_set_cookie = _extract_token_from_set_cookie_values(step6_set_cookie_values)
        step6_session_set_cookie_raw = _extract_session_set_cookie_raw(step6_set_cookie_values)
        # LOGGER.info("step6 Set-Cookie 全量条目如下（count=%s）:", len(step6_set_cookie_values))
        # if len(step6_set_cookie_values) == 0:
        #     LOGGER.warning(
        #         "step6 未收到任何 Set-Cookie。该请求可能成功，但服务端本次未下发 cookie（不一定是失败）。"
        #     )
        # for idx, set_cookie_item in enumerate(step6_set_cookie_values, start=1):
        #     LOGGER.info("step6 Set-Cookie[%s]: %s", idx, set_cookie_item)
        record_token("step6", "set-cookie.__Secure-next-auth.session-token", step6_session_from_set_cookie)
        record_token("step6", "set-cookie.__Secure-next-auth.session-token.raw", step6_session_set_cookie_raw)
        record_cookie_state("step6 完成后", capture_tokens=True)

    account.cookie = current_cookie()
    account.final_session_token = extract_session_token_from_cookie_header(account.cookie) or ""
    account.final_session_set_cookie_raw = step6_session_set_cookie_raw or ""
    if not step6_session_from_set_cookie and account.final_session_token:
        LOGGER.info("step6 未下发 session-token，沿用当前 Cookie 中已有 session-token。")
    if not account.final_session_token:
        account.final_session_token = step6_session_from_set_cookie or ""
    if not account.final_session_token:
        LOGGER.warning("最终仍未获取到 __Secure-next-auth.session-token，请重点检查 step5/step6 链路与服务端会话状态。")
    record_token("final", "__Secure-next-auth.session-token", account.final_session_token)
    record_token("final", "__Secure-next-auth.session-token.raw", account.final_session_set_cookie_raw)
    account.observed_tokens = observed_tokens

    if test_at_after_refresh:
        st_token = account.final_session_token
        record_token("step7", "candidate-st-for-st_to_at:__Secure-next-auth.session-token", st_token)
        _log_step_intro(
            step_no=7,
            title="ST->AT + AT 有效性测试",
            purpose="先用 ST 调 /api/auth/session 换取 AT，再用 AT 调 /v1/credits 验证可用性。",
            request_brief=f"GET {BASE_URL}/api/auth/session(Cookie=ST) -> GET {FLOW_API_BASE_URL}/credits(Bearer=AT)",
            expected_output="拿到 access_token/expires，且 credits 接口返回成功。",
            failure_impact="ST 无法换 AT 或换出的 AT 不可用，后续需要走其他恢复路径。",
        )
        st_to_at_result = _st_to_at_via_auth_session(
            st_token,
            proxy_url=proxy_url,
            connect_timeout=connect_timeout,
            timeout=timeout,
        )
        if not st_to_at_result.get("success"):
            _log_warning("step7 ST->AT 失败: %s", st_to_at_result.get("error"))
        else:
            at_token = st_to_at_result.get("access_token") or ""
            record_token("step7", "st_to_at.access_token", at_token)
            LOGGER.info("step7 ST->AT 成功: expires=%s", st_to_at_result.get("expires"))
            verify_result = _verify_at_via_get_credits(
                at_token,
                proxy_url=proxy_url,
                connect_timeout=connect_timeout,
                timeout=timeout,
            )
            if verify_result["success"]:
                LOGGER.info(
                    "step7 完成: AT 验证成功, credits=%s, tier=%s",
                    verify_result.get("credits"),
                    verify_result.get("userPaygateTier"),
                )
            else:
                _log_warning("step7 AT 验证失败: %s", verify_result.get("error"))

    LOGGER.info("========== reAuth 功能说明（结束） ==========")
    LOGGER.info("最终Cookie全文: %s", account.cookie or "<空Cookie>")
    LOGGER.info("最终SessionToken(完整): %s", account.final_session_token)
    LOGGER.info("最终Session Set-Cookie原文(完整): %s", account.final_session_set_cookie_raw or "<空>")
    # LOGGER.info("过程采集到的全部Token如下（按时间顺序）:")
    # for idx, item in enumerate(observed_tokens, start=1):
    #     LOGGER.info("  [%s] stage=%s source=%s token=%s", idx, item["stage"], item["source"], item["token"] or "<空>")
    LOGGER.info("==========================================")
    return account.cookie


def _main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Flow cookies before re-login.")
    parser.add_argument("--project-id", required=False, help="Flow project id")
    parser.add_argument("--cookie", required=True, help="Current Cookie header string")
    parser.add_argument("--cookie-file", default=None, help="Optional cookie string for step4")
    parser.add_argument("--ua", default=DEFAULT_USER_AGENT, help="User-Agent")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--connect-timeout", type=int, default=10, help="Connect timeout seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for each HTTP step")
    parser.add_argument("--proxy", default=None, help="Optional proxy URL")
    parser.add_argument("--local-proxy-port", type=int, default=None, help="Use local HTTP proxy port, e.g. 6987")
    parser.add_argument("--test-at", action="store_true", help="After refresh, test candidate token via /v1/credits once")
    parser.add_argument("--verify-only", action="store_true", help="Only verify input cookie token via /v1/credits once")
    parser.add_argument("--strict-java-flow", action="store_true", help="Strict 1:1 Java flow: cookie update only in step3/5/6, step4 uses cookie_file, step5 uses location header only")
    parser.add_argument("--test-browser-with-cookie", action="store_true", help="After refresh (or verify-only), open browser with current cookie and visit target url for manual validation")
    parser.add_argument("--browser-url", default=DEFAULT_BROWSER_TEST_URL, help="Browser validation target url")
    parser.add_argument("--browser-headless", action="store_true", help="Run browser validation in headless mode")
    parser.add_argument("--browser-wait-seconds", type=int, default=45, help="Seconds to keep browser open for manual checking")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.verify_only and not args.project_id:
        parser.error("--project-id is required unless --verify-only is enabled")

    proxy_value = args.proxy
    if args.local_proxy_port:
        proxy_value = f"127.0.0.1:{args.local_proxy_port}"
    effective_proxy = normalize_proxy_url(proxy_value)
    if effective_proxy:
        LOGGER.info("Using proxy: %s", effective_proxy)

    try:
        if args.verify_only:
            output = verify_input_cookie_only(
                args.cookie,
                proxy_url=effective_proxy,
                connect_timeout=args.connect_timeout,
                timeout=args.timeout,
            )
            if args.test_browser_with_cookie:
                output["browser_cookie_test"] = open_browser_with_cookie_for_validation(
                    args.cookie,
                    target_url=args.browser_url,
                    user_agent=args.ua,
                    proxy_url=effective_proxy,
                    headless=args.browser_headless,
                    wait_seconds=args.browser_wait_seconds,
                )
            print(json.dumps(output, ensure_ascii=False))
            return

        account = ReAuthAccount(
            project_id=args.project_id or "",
            cookie=args.cookie,
            cookie_file=args.cookie_file,
        )
        refreshed_cookie = refresh_cookie_before_relogin(
            account=account,
            user_agent=args.ua,
            timeout=args.timeout,
            connect_timeout=args.connect_timeout,
            max_retries=max(1, args.retries),
            proxy_url=effective_proxy,
            test_at_after_refresh=args.test_at,
            strict_java_flow=args.strict_java_flow,
        )
        output: Dict[str, Any] = {"cookie": refreshed_cookie}
        output["strict_java_flow"] = args.strict_java_flow
        output["observed_tokens"] = account.observed_tokens
        output["final_session_token"] = account.final_session_token
        output["final_session_set_cookie_raw"] = account.final_session_set_cookie_raw
        if args.test_at:
            st_to_at_result = _st_to_at_via_auth_session(
                account.final_session_token,
                proxy_url=effective_proxy,
                connect_timeout=args.connect_timeout,
                timeout=args.timeout,
            )
            output["st_to_at"] = st_to_at_result
            if st_to_at_result.get("success"):
                candidate_token = st_to_at_result.get("access_token") or ""
                output["at_test"] = _verify_at_via_get_credits(
                    candidate_token,
                    proxy_url=effective_proxy,
                    connect_timeout=args.connect_timeout,
                    timeout=args.timeout,
                )
            else:
                output["at_test"] = {
                    "success": False,
                    "credits": None,
                    "userPaygateTier": None,
                    "error": f"ST->AT failed: {st_to_at_result.get('error')}",
                }
        if args.test_browser_with_cookie:
            output["browser_cookie_test"] = open_browser_with_cookie_for_validation(
                account.cookie,
                target_url=args.browser_url,
                user_agent=args.ua,
                proxy_url=effective_proxy,
                headless=args.browser_headless,
                wait_seconds=args.browser_wait_seconds,
            )
        # print(json.dumps(output, ensure_ascii=False))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        _log_warning("reAuth failed: %s", e)
        LOGGER.error("reAuth failed: %s", e)
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
