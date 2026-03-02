"""Proxy management module"""
import json
from pathlib import Path
from typing import Optional, Tuple
import re
from ..core.database import Database
from ..core.models import ProxyConfig
from ..core.logger import debug_logger


class ProxyManager:
    """Proxy configuration manager"""

    def __init__(self, db: Database):
        self.db = db
        self.proxy_pool_service = None
        self._tmp_token_dir = Path(__file__).resolve().parents[2] / "tmp" / "Token"

    def set_proxy_pool_service(self, service) -> None:
        """Inject proxy pool service lazily to avoid import coupling."""
        self.proxy_pool_service = service

    def _parse_proxy_line(self, line: str) -> Optional[str]:
        """将用户输入代理转换为标准 URL 格式。

        支持格式：
        - http://user:pass@host:port
        - https://user:pass@host:port
        - socks5://user:pass@host:port
        - socks5h://user:pass@host:port
        - socks5://host:port:user:pass
        - st5 host:port:user:pass
        - host:port
        - host:port:user:pass
        """
        if not line:
            return None

        line = line.strip()
        if not line:
            return None

        # st5 host:port:user:pass
        st5_match = re.match(r"^st5\s+(.+)$", line, re.IGNORECASE)
        if st5_match:
            rest = st5_match.group(1).strip()
            if "@" in rest:
                return f"socks5://{rest}"
            parts = rest.split(":")
            if len(parts) >= 4 and parts[1].isdigit():
                host = parts[0]
                port = parts[1]
                username = parts[2]
                password = ":".join(parts[3:])
                return f"socks5://{username}:{password}@{host}:{port}"
            return None

        # 协议前缀格式
        if line.startswith(("http://", "https://", "socks5://", "socks5h://")):
            # socks5h 统一转 socks5，便于后续处理
            if line.startswith("socks5h://"):
                line = "socks5://" + line[len("socks5h://"):]

            # 已是标准 user:pass@host:port（或 host:port）
            if "@" in line:
                return line

            # 兼容 protocol://host:port:user:pass
            try:
                protocol_end = line.index("://") + 3
                protocol = line[:protocol_end]
                rest = line[protocol_end:]
                parts = rest.split(":")
                if len(parts) >= 4 and parts[1].isdigit():
                    host = parts[0]
                    port = parts[1]
                    username = parts[2]
                    password = ":".join(parts[3:])
                    return f"{protocol}{username}:{password}@{host}:{port}"
                if len(parts) == 2 and parts[1].isdigit():
                    return line
            except Exception:
                return None
            return None

        # 无协议，带 @：默认按 http 处理
        if "@" in line:
            return f"http://{line}"

        # 无协议，按冒号数量判断
        parts = line.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            # host:port
            return f"http://{parts[0]}:{parts[1]}"

        if len(parts) >= 4 and parts[1].isdigit():
            # host:port:user:pass
            host = parts[0]
            port = parts[1]
            username = parts[2]
            password = ":".join(parts[3:])
            return f"http://{username}:{password}@{host}:{port}"

        return None

    def normalize_proxy_url(self, proxy_url: Optional[str]) -> Optional[str]:
        """标准化代理地址，空值返回 None，非法格式抛 ValueError。"""
        if proxy_url is None:
            return None

        raw = proxy_url.strip()
        if not raw:
            return None

        parsed = self._parse_proxy_line(raw)
        if not parsed:
            raise ValueError(
                "代理地址格式错误，支持示例："
                "http://user:pass@host:port / "
                "socks5://user:pass@host:port / "
                "host:port:user:pass / st5 host:port:user:pass"
            )
        return parsed

    async def get_proxy_url(self) -> Optional[str]:
        """兼容旧调用：返回请求代理地址"""
        return await self.get_request_proxy_url()

    async def get_request_proxy_url(self) -> Optional[str]:
        """Get request proxy URL if enabled, otherwise return None"""
        config = await self.db.get_proxy_config()
        if config and config.enabled and config.proxy_url:
            return config.proxy_url
        return None

    async def get_media_proxy_url(self) -> Optional[str]:
        """Get media upload/download proxy URL, fallback to request proxy"""
        config = await self.db.get_proxy_config()
        if config and config.media_proxy_enabled and config.media_proxy_url:
            return config.media_proxy_url
        return await self.get_request_proxy_url()

    @staticmethod
    def _extract_email_from_token_file(path: Path) -> Optional[str]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

        email = str((payload or {}).get("email") or "").strip()
        if "@" in email:
            return email

        stem = path.stem
        if "_" in stem:
            prefix, suffix = stem.rsplit("_", 1)
            if suffix.isdigit():
                stem = prefix
        guessed = stem.replace("_at_", "@").strip()
        if "@" in guessed:
            return guessed
        return None

    def _resolve_credential_filename_by_email(self, email: str) -> Optional[str]:
        mail = str(email or "").strip().lower()
        if not mail:
            return None
        if not self._tmp_token_dir.exists() or not self._tmp_token_dir.is_dir():
            return None

        matched = []
        for path in self._tmp_token_dir.glob("*.json"):
            file_email = str(self._extract_email_from_token_file(path) or "").strip().lower()
            if file_email == mail:
                matched.append(path)

        if not matched:
            return None

        matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matched[0].name

    async def _resolve_credential_name_for_token(
        self,
        *,
        st_token: Optional[str] = None,
        at_token: Optional[str] = None,
    ) -> Optional[str]:
        token = None
        st = str(st_token or "").strip()
        at = str(at_token or "").strip()
        if st:
            token = await self.db.get_token_by_st(st)
        if token is None and at:
            token = await self.db.get_token_by_at(at)

        if token is None:
            return None

        email = str(getattr(token, "email", "") or "").strip()
        if not email:
            return None

        # Prefer real credential filename so manual bindings in proxy pool work.
        credential_filename = self._resolve_credential_filename_by_email(email)
        if credential_filename:
            return credential_filename

        # Fallback: use email as stable binding key.
        return email

    async def select_proxy_url(
        self,
        *,
        st_token: Optional[str] = None,
        at_token: Optional[str] = None,
        use_media_proxy: bool = False,
    ) -> Optional[str]:
        """Return selected proxy URL only (compat)."""
        proxy_url, _ = await self.select_proxy_with_source(
            st_token=st_token,
            at_token=at_token,
            use_media_proxy=use_media_proxy,
        )
        return proxy_url

    async def select_proxy_with_source(
        self,
        *,
        st_token: Optional[str] = None,
        at_token: Optional[str] = None,
        use_media_proxy: bool = False,
    ) -> Tuple[Optional[str], str]:
        """
        Request proxy priority:
        1) System proxy config (request/media proxy)
        2) Proxy pool assigned proxy (credential-bound first, then any available)
        """
        # Priority 1: system proxy config
        system_proxy = await (
            self.get_media_proxy_url() if use_media_proxy else self.get_request_proxy_url()
        )
        if system_proxy:
            return system_proxy, "system"

        # Priority 2: proxy pool
        service = self.proxy_pool_service
        if service is None:
            return None, "direct"

        try:
            credential_name = await self._resolve_credential_name_for_token(
                st_token=st_token,
                at_token=at_token,
            )
            if credential_name:
                proxy_url = await service.get_proxy_url_for_credential(
                    credential_name=credential_name,
                    mode="antigravity",
                    force_rebind=False,
                )
                if proxy_url:
                    return proxy_url, "proxy_pool"
            proxy_url = await service.get_any_proxy_url()
            if proxy_url:
                return proxy_url, "proxy_pool"
            return None, "direct"
        except Exception as e:
            debug_logger.log_warning(f"[ProxyPool] select_proxy_url failed: {e}")
            return None, "direct"

    async def update_proxy_config(
        self,
        enabled: bool,
        proxy_url: Optional[str],
        media_proxy_enabled: Optional[bool] = None,
        media_proxy_url: Optional[str] = None
    ):
        """Update proxy configuration"""
        normalized_proxy_url = self.normalize_proxy_url(proxy_url)
        normalized_media_proxy_url = self.normalize_proxy_url(media_proxy_url)

        await self.db.update_proxy_config(
            enabled=enabled,
            proxy_url=normalized_proxy_url,
            media_proxy_enabled=media_proxy_enabled,
            media_proxy_url=normalized_media_proxy_url
        )

    async def get_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration"""
        return await self.db.get_proxy_config()
