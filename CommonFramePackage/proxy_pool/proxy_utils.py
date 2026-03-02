"""Proxy parsing, URL building and health-test helpers."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
from urllib.parse import quote

import httpx

DEFAULT_TEST_URLS: Tuple[str, ...] = (
    "https://httpbin.org/ip",
    "https://api.ipify.org?format=json",
)

_SCHEME_RE = re.compile(r"^(?P<scheme>https?|socks5h?)://", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass(frozen=True)
class ProxySpec:
    scheme: str
    host: str
    port: int
    username: str
    password: str

    @property
    def proxy_url(self) -> str:
        return build_proxy_url(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            scheme=self.scheme,
        )


def _parse_kv_style_proxy(raw: str) -> Optional[dict]:
    if " " not in raw:
        return None

    tokens = [t.strip() for t in raw.split() if t.strip()]
    if not any(
        t.lower().startswith(
            (
                "host:",
                "port:",
                "username:",
                "password:",
                "user:",
                "pass:",
                "pwd:",
                "scheme:",
            )
        )
        for t in tokens
    ):
        return None

    kv = {}
    for token in tokens:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        key = (key or "").strip().lower()
        value = (value or "").strip()
        if key:
            kv[key] = value
    return kv


def parse_proxy_line(line: str, default_scheme: str = "http") -> Optional[ProxySpec]:
    raw = (line or "").strip()
    if not raw or raw.startswith("#"):
        return None

    scheme = (default_scheme or "http").strip().lower()
    m = _SCHEME_RE.match(raw)
    if m:
        scheme = m.group("scheme").lower()
        raw = raw[m.end() :]

    kv = _parse_kv_style_proxy(raw)
    if kv is not None:
        host = (kv.get("host") or "").strip()
        port_str = (kv.get("port") or "").strip()
        username = (kv.get("username") or kv.get("user") or "").strip()
        password = (kv.get("password") or kv.get("pass") or kv.get("pwd") or "").strip()
        if kv.get("scheme"):
            scheme = str(kv.get("scheme")).strip().lower() or scheme
    else:
        parts = raw.split(":")
        if len(parts) < 4:
            raise ValueError(
                "invalid proxy line: expected host:port:user:pass or kv style host:.. port:.. username:.. password:.."
            )
        host = parts[0].strip()
        port_str = parts[1].strip()
        username = parts[2].strip()
        password = ":".join(parts[3:]).strip()

    if scheme not in ("http", "https", "socks5", "socks5h"):
        raise ValueError(f"unsupported scheme: {scheme}")
    if not host:
        raise ValueError("host is empty")
    if not port_str.isdigit():
        raise ValueError(f"port is not a number: {port_str!r}")
    port = int(port_str)
    if port <= 0 or port > 65535:
        raise ValueError(f"port out of range: {port}")
    if not username:
        raise ValueError("username is empty")
    if password == "":
        raise ValueError("password is empty")

    return ProxySpec(
        scheme=scheme,
        host=host,
        port=port,
        username=username,
        password=password,
    )


def build_proxy_url(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    scheme: str = "http",
) -> str:
    safe_user = quote(str(username), safe="")
    safe_pass = quote(str(password), safe="")
    return f"{scheme}://{safe_user}:{safe_pass}@{host}:{int(port)}"


def _extract_ip_from_payload(payload: object) -> Optional[str]:
    if isinstance(payload, dict):
        ip = payload.get("ip")
        if isinstance(ip, str) and ip.strip():
            return ip.strip()
        origin = payload.get("origin")
        if isinstance(origin, str) and origin.strip():
            m = _IPV4_RE.search(origin)
            if m:
                return m.group(0)
    return None


def extract_ip_from_message(msg: str) -> Optional[str]:
    if not msg:
        return None
    brace = msg.find("{")
    if brace >= 0:
        try:
            payload = json.loads(msg[brace:])
            ip = _extract_ip_from_payload(payload)
            if ip:
                return ip
        except Exception:
            pass
    m = _IPV4_RE.search(msg)
    return m.group(0) if m else None


async def _fetch_with_proxy(
    *,
    url: str,
    proxy_url: str,
    timeout_s: float,
) -> Tuple[bool, str, Optional[str]]:
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout_s, proxy=proxy_url, follow_redirects=True) as client:
        response = await client.get(url)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        content_type = (response.headers.get("content-type") or "").lower()
        payload_text: str
        ip: Optional[str] = None
        if "application/json" in content_type:
            payload = response.json()
            ip = _extract_ip_from_payload(payload)
            payload_text = json.dumps(payload, ensure_ascii=False)
        else:
            body = (response.text or "").strip()
            payload_text = body[:300]
            ip = extract_ip_from_message(payload_text)

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code} ({elapsed_ms}ms): {payload_text}", None
        return True, f"OK ({elapsed_ms}ms) {payload_text}", ip


async def test_proxy_once(
    proxy: ProxySpec,
    *,
    timeout_s: float = 8.0,
    test_urls: Optional[Iterable[str]] = None,
) -> Tuple[bool, str, str, Optional[str]]:
    urls = [u for u in (test_urls or DEFAULT_TEST_URLS) if (u or "").strip()]
    if not urls:
        urls = list(DEFAULT_TEST_URLS)

    last_msg = "no test url"
    for url in urls:
        try:
            ok, msg, ip = await _fetch_with_proxy(
                url=url,
                proxy_url=proxy.proxy_url,
                timeout_s=float(timeout_s),
            )
            if ok:
                return True, url, msg, ip
            last_msg = f"{url} -> {msg}"
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            last_msg = f"{url} -> {type(exc).__name__}: {exc}"
        except Exception as exc:
            last_msg = f"{url} -> unexpected {type(exc).__name__}: {exc}"
    return False, "", last_msg, None
