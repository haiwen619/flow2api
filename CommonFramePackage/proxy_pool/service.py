"""Business service for proxy pool backend."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .proxy_utils import (
    ProxySpec,
    build_proxy_url,
    parse_proxy_line,
    test_proxy_once,
)
from .repository import ProxyPoolRepository


class ProxyPoolService:
    def __init__(self, repo: ProxyPoolRepository) -> None:
        self.repo = repo

    async def initialize(self) -> None:
        await self.repo.initialize()

    async def add_proxy(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return await self.repo.upsert_proxy(
            host=host,
            port=port,
            username=username,
            password=password,
            tags=tags or [],
        )

    async def list_proxies(
        self,
        *,
        offset: int,
        limit: int,
        search: Optional[str],
        host: Optional[str],
    ) -> Dict[str, Any]:
        return await self.repo.list_proxies(
            offset=offset,
            limit=limit,
            search=search,
            host=host,
        )

    async def update_proxy(
        self,
        *,
        proxy_key: str,
        host: Optional[str],
        port: Optional[int],
        username: Optional[str],
        password: Optional[str],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return await self.repo.update_proxy(
            proxy_key=proxy_key,
            host=host,
            port=port,
            username=username,
            password=password,
            tags=tags,
        )

    async def delete_proxy(self, *, proxy_key: str) -> None:
        await self.repo.delete_proxy(proxy_key=proxy_key)

    async def bulk_import(
        self,
        *,
        text: str,
        default_scheme: str = "http",
        tags: Optional[List[str]] = None,
        max_lines: int = 5000,
    ) -> Dict[str, Any]:
        lines = (text or "").splitlines()
        if len(lines) > max_lines:
            raise ValueError(f"too many lines, max={max_lines}")

        imported = 0
        fail_items: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines, start=1):
            raw = (line or "").strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                spec = parse_proxy_line(raw, default_scheme=default_scheme or "http")
                if spec is None:
                    continue
                await self.repo.upsert_proxy(
                    host=spec.host,
                    port=spec.port,
                    username=spec.username,
                    password=spec.password,
                    tags=tags or [],
                )
                imported += 1
            except Exception as exc:
                fail_items.append({"line": idx, "raw": line, "error": str(exc)})

        return {
            "imported": imported,
            "failed": len(fail_items),
            "fail_items": fail_items[:50],
        }

    async def test_proxy(
        self,
        *,
        proxy_key: str,
        timeout_s: float = 8.0,
        test_urls: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        secret = await self.repo.get_proxy_secret(proxy_key=proxy_key)
        spec = ProxySpec(
            scheme="http",
            host=secret["host"],
            port=int(secret["port"]),
            username=secret["username"],
            password=secret["password"],
        )

        ok, used_url, msg, ip = await test_proxy_once(
            spec,
            timeout_s=float(timeout_s or 8.0),
            test_urls=test_urls,
        )
        full_msg = f"{used_url} -> {msg}" if ok and used_url else msg
        await self.repo.set_test_result(
            proxy_key=proxy_key,
            ok=ok,
            ip=ip,
            msg=full_msg,
        )
        return {"ok": bool(ok), "ip": ip, "msg": full_msg}

    async def bind_proxy(
        self,
        *,
        proxy_key: str,
        credential_name: str,
        mode: str = "antigravity",
        force: bool = False,
    ) -> Dict[str, Any]:
        return await self.repo.bind_proxy_to_credential(
            proxy_key=proxy_key,
            credential_name=credential_name,
            mode=mode,
            force=force,
        )

    async def unbind_proxy(self, *, proxy_key: str) -> Dict[str, Any]:
        return await self.repo.unbind_proxy(proxy_key=proxy_key)

    async def set_disabled(self, *, proxy_key: str, disabled: bool) -> Dict[str, Any]:
        return await self.repo.set_proxy_disabled(proxy_key=proxy_key, disabled=disabled)

    async def get_any_proxy_url(self) -> Optional[str]:
        result = await self.repo.list_proxies(offset=0, limit=50, search=None, host=None)
        items = (result or {}).get("items") or []
        if not items:
            return None

        ok_items = [x for x in items if not x.get("disabled") and x.get("last_test_ok") in (1, True)]
        candidates = ok_items if ok_items else [x for x in items if not x.get("disabled")]
        if not candidates:
            return None

        picked = candidates[0]
        secret = await self.repo.get_proxy_secret(proxy_key=str(picked["proxy_key"]))
        return build_proxy_url(
            host=secret["host"],
            port=int(secret["port"]),
            username=secret["username"],
            password=secret["password"],
            scheme="http",
        )

    async def get_proxy_url_for_credential(
        self,
        *,
        credential_name: str,
        mode: str = "antigravity",
        force_rebind: bool = False,
    ) -> Optional[str]:
        item = await self.repo.get_or_bind_proxy_for_credential(
            credential_name=credential_name,
            mode=mode,
            force_rebind=force_rebind,
        )
        if not item:
            return None
        return build_proxy_url(
            host=str(item.get("host") or ""),
            port=int(item.get("port") or 0),
            username=str(item.get("username") or ""),
            password=str(item.get("password") or ""),
            scheme="http",
        )
