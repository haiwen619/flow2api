"""Configuration management for Flow2API"""
import ipaddress
import tomli
import re
from urllib import request as urllib_request
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Application configuration"""

    def __init__(self):
        self._config_path = Path(__file__).parent.parent.parent / "config" / "setting.toml"
        self._config = self._load_config()
        self._admin_username: Optional[str] = None
        self._admin_password: Optional[str] = None
        self._runtime_server_host_override: Optional[str] = None
        self._runtime_server_mode_override: Optional[str] = None
        self._detected_public_ip: Optional[str] = None
        self._server_auto_detected: bool = False

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from setting.toml"""
        with open(self._config_path, "rb") as f:
            return tomli.load(f)

    def reload_config(self):
        """Reload configuration from file"""
        self._config = self._load_config()

    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._config

    @property
    def admin_username(self) -> str:
        # If admin_username is set from database, use it; otherwise fall back to config file
        if self._admin_username is not None:
            return self._admin_username
        return self._config["global"]["admin_username"]

    @admin_username.setter
    def admin_username(self, value: str):
        self._admin_username = value
        self._config["global"]["admin_username"] = value

    def set_admin_username_from_db(self, username: str):
        """Set admin username from database"""
        self._admin_username = username

    # Flow2API specific properties
    @property
    def flow_labs_base_url(self) -> str:
        """Google Labs base URL for project management"""
        return self._config["flow"]["labs_base_url"]

    @property
    def flow_api_base_url(self) -> str:
        """Google AI Sandbox API base URL for generation"""
        return self._config["flow"]["api_base_url"]

    @property
    def flow_timeout(self) -> int:
        return self._config["flow"]["timeout"]

    @property
    def flow_max_retries(self) -> int:
        return self._config["flow"]["max_retries"]

    @property
    def enable_reauth_refresh(self) -> bool:
        """Whether AT refresh should try reAuth HTTP recovery path."""
        # Default to True so AT auto-refresh can include reAuth cookie/session renewal
        # even if legacy setting.toml does not yet contain this key.
        return bool(self._config.get("flow", {}).get("enable_reauth_refresh", True))

    def set_enable_reauth_refresh(self, enabled: bool):
        """Set reAuth HTTP recovery switch for AT refresh path."""
        if "flow" not in self._config:
            self._config["flow"] = {}
        self._config["flow"]["enable_reauth_refresh"] = bool(enabled)

    @property
    def reauth_cookie_invalid_auto_login_enabled(self) -> bool:
        """Whether cookie-invalid(interaction_required) should trigger account-pool auto login."""
        return bool(
            self._config.get("flow", {}).get(
                "reauth_cookie_invalid_auto_login_enabled",
                False,
            )
        )

    def set_reauth_cookie_invalid_auto_login_enabled(self, enabled: bool):
        """Set account-pool auto login switch for cookie-invalid reAuth failures."""
        if "flow" not in self._config:
            self._config["flow"] = {}
        self._config["flow"]["reauth_cookie_invalid_auto_login_enabled"] = bool(enabled)

    def update_flow_switches(
        self,
        *,
        reauth_cookie_invalid_auto_login_enabled: Optional[bool] = None,
        enable_reauth_refresh: Optional[bool] = None,
    ):
        """Persist mutable flow switches into setting.toml and reload memory config."""
        if (
            reauth_cookie_invalid_auto_login_enabled is None
            and enable_reauth_refresh is None
        ):
            return

        content = self._config_path.read_text(encoding="utf-8")

        if reauth_cookie_invalid_auto_login_enabled is not None:
            content = self._upsert_toml_key_in_section(
                content=content,
                section="flow",
                key="reauth_cookie_invalid_auto_login_enabled",
                value_literal=(
                    "true" if bool(reauth_cookie_invalid_auto_login_enabled) else "false"
                ),
            )

        if enable_reauth_refresh is not None:
            content = self._upsert_toml_key_in_section(
                content=content,
                section="flow",
                key="enable_reauth_refresh",
                value_literal=("true" if bool(enable_reauth_refresh) else "false"),
            )

        self._config_path.write_text(content, encoding="utf-8")
        self.reload_config()

    @property
    def poll_interval(self) -> float:
        return self._config["flow"]["poll_interval"]

    @property
    def max_poll_attempts(self) -> int:
        return self._config["flow"]["max_poll_attempts"]

    @property
    def server_host(self) -> str:
        if self._runtime_server_host_override is not None:
            return self._runtime_server_host_override
        return self._config["server"]["host"]

    @property
    def server_port(self) -> int:
        return self._config["server"]["port"]

    @property
    def configured_server_host(self) -> str:
        return str(self._config.get("server", {}).get("host", "") or "").strip()

    @property
    def default_server_public_ip(self) -> str:
        return str(self._config.get("server", {}).get("default_public_ip", "") or "").strip()

    @property
    def detected_public_ip(self) -> str:
        return str(self._detected_public_ip or "").strip()

    @property
    def server_auto_detected(self) -> bool:
        return bool(self._server_auto_detected)

    @property
    def debug_enabled(self) -> bool:
        return self._config.get("debug", {}).get("enabled", False)

    @property
    def debug_log_requests(self) -> bool:
        return self._config.get("debug", {}).get("log_requests", True)

    @property
    def debug_log_responses(self) -> bool:
        return self._config.get("debug", {}).get("log_responses", True)

    @property
    def debug_mask_token(self) -> bool:
        return self._config.get("debug", {}).get("mask_token", True)

    # Mutable properties for runtime updates
    @property
    def api_key(self) -> str:
        return self._config["global"]["api_key"]

    @api_key.setter
    def api_key(self, value: str):
        self._config["global"]["api_key"] = value

    @property
    def admin_password(self) -> str:
        # If admin_password is set from database, use it; otherwise fall back to config file
        if self._admin_password is not None:
            return self._admin_password
        return self._config["global"]["admin_password"]

    @admin_password.setter
    def admin_password(self, value: str):
        self._admin_password = value
        self._config["global"]["admin_password"] = value

    def set_admin_password_from_db(self, password: str):
        """Set admin password from database"""
        self._admin_password = password

    def set_debug_enabled(self, enabled: bool):
        """Set debug mode enabled/disabled"""
        if "debug" not in self._config:
            self._config["debug"] = {}
        self._config["debug"]["enabled"] = enabled

    @property
    def image_timeout(self) -> int:
        """Get image generation timeout in seconds"""
        return self._config.get("generation", {}).get("image_timeout", 300)

    def set_image_timeout(self, timeout: int):
        """Set image generation timeout in seconds"""
        if "generation" not in self._config:
            self._config["generation"] = {}
        self._config["generation"]["image_timeout"] = timeout

    @property
    def video_timeout(self) -> int:
        """Get video generation timeout in seconds"""
        return self._config.get("generation", {}).get("video_timeout", 1500)

    def set_video_timeout(self, timeout: int):
        """Set video generation timeout in seconds"""
        if "generation" not in self._config:
            self._config["generation"] = {}
        self._config["generation"]["video_timeout"] = timeout

    @property
    def upsample_timeout(self) -> int:
        """Get upsample (4K/2K) timeout in seconds"""
        return self._config.get("generation", {}).get("upsample_timeout", 300)

    def set_upsample_timeout(self, timeout: int):
        """Set upsample (4K/2K) timeout in seconds"""
        if "generation" not in self._config:
            self._config["generation"] = {}
        self._config["generation"]["upsample_timeout"] = timeout

    # Cache configuration
    @property
    def cache_enabled(self) -> bool:
        """Get cache enabled status"""
        return self._config.get("cache", {}).get("enabled", False)

    def set_cache_enabled(self, enabled: bool):
        """Set cache enabled status"""
        if "cache" not in self._config:
            self._config["cache"] = {}
        self._config["cache"]["enabled"] = enabled

    @property
    def cache_timeout(self) -> int:
        """Get cache timeout in seconds"""
        return self._config.get("cache", {}).get("timeout", 7200)

    def set_cache_timeout(self, timeout: int):
        """Set cache timeout in seconds"""
        if "cache" not in self._config:
            self._config["cache"] = {}
        self._config["cache"]["timeout"] = timeout

    @property
    def cache_base_url(self) -> str:
        """Get cache base URL"""
        return self._config.get("cache", {}).get("base_url", "")

    def set_cache_base_url(self, base_url: str):
        """Set cache base URL"""
        if "cache" not in self._config:
            self._config["cache"] = {}
        self._config["cache"]["base_url"] = base_url

    # Captcha configuration
    @property
    def captcha_method(self) -> str:
        """Get captcha method"""
        return self._config.get("captcha", {}).get("captcha_method", "yescaptcha")

    def set_captcha_method(self, method: str):
        """Set captcha method"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["captcha_method"] = method

    @property
    def yescaptcha_api_key(self) -> str:
        """Get YesCaptcha API key"""
        return self._config.get("captcha", {}).get("yescaptcha_api_key", "")

    def set_yescaptcha_api_key(self, api_key: str):
        """Set YesCaptcha API key"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["yescaptcha_api_key"] = api_key

    @property
    def yescaptcha_base_url(self) -> str:
        """Get YesCaptcha base URL"""
        return self._config.get("captcha", {}).get("yescaptcha_base_url", "https://api.yescaptcha.com")

    def set_yescaptcha_base_url(self, base_url: str):
        """Set YesCaptcha base URL"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["yescaptcha_base_url"] = base_url

    @property
    def capmonster_api_key(self) -> str:
        """Get CapMonster API key"""
        return self._config.get("captcha", {}).get("capmonster_api_key", "")

    def set_capmonster_api_key(self, api_key: str):
        """Set CapMonster API key"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["capmonster_api_key"] = api_key

    @property
    def capmonster_base_url(self) -> str:
        """Get CapMonster base URL"""
        return self._config.get("captcha", {}).get("capmonster_base_url", "https://api.capmonster.cloud")

    def set_capmonster_base_url(self, base_url: str):
        """Set CapMonster base URL"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["capmonster_base_url"] = base_url

    @property
    def ezcaptcha_api_key(self) -> str:
        """Get EzCaptcha API key"""
        return self._config.get("captcha", {}).get("ezcaptcha_api_key", "")

    def set_ezcaptcha_api_key(self, api_key: str):
        """Set EzCaptcha API key"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["ezcaptcha_api_key"] = api_key

    @property
    def ezcaptcha_base_url(self) -> str:
        """Get EzCaptcha base URL"""
        return self._config.get("captcha", {}).get("ezcaptcha_base_url", "https://api.ez-captcha.com")

    def set_ezcaptcha_base_url(self, base_url: str):
        """Set EzCaptcha base URL"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["ezcaptcha_base_url"] = base_url

    @property
    def capsolver_api_key(self) -> str:
        """Get CapSolver API key"""
        return self._config.get("captcha", {}).get("capsolver_api_key", "")

    def set_capsolver_api_key(self, api_key: str):
        """Set CapSolver API key"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["capsolver_api_key"] = api_key

    @property
    def capsolver_base_url(self) -> str:
        """Get CapSolver base URL"""
        return self._config.get("captcha", {}).get("capsolver_base_url", "https://api.capsolver.com")

    def set_capsolver_base_url(self, base_url: str):
        """Set CapSolver base URL"""
        if "captcha" not in self._config:
            self._config["captcha"] = {}
        self._config["captcha"]["capsolver_base_url"] = base_url

    def _upsert_toml_key_in_section(
        self,
        content: str,
        section: str,
        key: str,
        value_literal: str,
    ) -> str:
        """Insert or replace `key = value` in the target TOML section."""
        section_pattern = re.compile(rf"(?m)^\[{re.escape(section)}\]\s*$")
        section_match = section_pattern.search(content)

        # If section does not exist, append a new section to the tail.
        if not section_match:
            tail = content
            if tail and not tail.endswith("\n"):
                tail += "\n"
            if tail:
                tail += "\n"
            tail += f"[{section}]\n{key} = {value_literal}\n"
            return tail

        section_body_start = section_match.end()
        next_section = re.search(r"(?m)^\[[^\]]+\]\s*$", content[section_body_start:])
        section_body_end = (
            section_body_start + next_section.start()
            if next_section
            else len(content)
        )
        section_body = content[section_body_start:section_body_end]

        key_pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
        if key_pattern.search(section_body):
            section_body = key_pattern.sub(
                lambda m: f"{m.group(1)}{value_literal}",
                section_body,
                count=1,
            )
        else:
            if section_body and not section_body.endswith("\n"):
                section_body += "\n"
            section_body += f"{key} = {value_literal}\n"

        return (
            content[:section_body_start]
            + section_body
            + content[section_body_end:]
        )

    def get_server_mode(self) -> str:
        """Infer runtime mode by host value in [server]."""
        if self._runtime_server_mode_override in {"local", "server"}:
            return self._runtime_server_mode_override
        host = str(self._config.get("server", {}).get("host", "") or "").strip().lower()
        return "local" if host in {"127.0.0.1", "localhost"} else "server"

    def normalize_server_host_for_mode(
        self,
        *,
        mode: str,
        host: Optional[str],
        default_public_ip: Optional[str] = None,
    ) -> str:
        mode_value = str(mode or "").strip().lower()
        host_value = str(host or "").strip()
        if mode_value == "local":
            return host_value or "127.0.0.1"

        if mode_value != "server":
            return host_value

        if not host_value:
            return "0.0.0.0"

        host_lower = host_value.lower()
        if host_lower in {"0.0.0.0", "::", "[::]"}:
            return "0.0.0.0"

        normalized_host = self._normalize_ip_text(host_value)
        normalized_default = self._normalize_ip_text(
            default_public_ip if default_public_ip is not None else self.default_server_public_ip
        )
        normalized_detected = self._normalize_ip_text(self.detected_public_ip)

        if normalized_host and normalized_host in {normalized_default, normalized_detected}:
            return "0.0.0.0"

        return host_value

    @staticmethod
    def _normalize_ip_text(value: Optional[str]) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return str(ipaddress.ip_address(text))
        except Exception:
            return ""

    def detect_public_ip(self, timeout_seconds: float = 3.0) -> str:
        providers = [
            "https://api.ipify.org",
            "https://api64.ipify.org",
            "https://ifconfig.me/ip",
            "https://icanhazip.com",
        ]
        headers = {"User-Agent": "Flow2API/1.0"}

        for url in providers:
            try:
                req = urllib_request.Request(url, headers=headers, method="GET")
                with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8", errors="replace").strip()
                normalized = self._normalize_ip_text(raw)
                if normalized:
                    self._detected_public_ip = normalized
                    return normalized
            except Exception:
                continue

        self._detected_public_ip = ""
        return ""

    def bootstrap_runtime_server_mode(self) -> Dict[str, Any]:
        configured_host = self.configured_server_host
        default_public_ip = self.default_server_public_ip
        detected_public_ip = self.detect_public_ip()

        self._runtime_server_host_override = None
        self._runtime_server_mode_override = None
        self._server_auto_detected = False

        normalized_default = self._normalize_ip_text(default_public_ip)
        normalized_detected = self._normalize_ip_text(detected_public_ip)
        matched_server = bool(
            normalized_default
            and normalized_detected
            and normalized_default == normalized_detected
        )

        if matched_server:
            self._runtime_server_host_override = normalized_default
            self._runtime_server_mode_override = "server"
            self._server_auto_detected = True

        return {
            "matched_server": matched_server,
            "configured_host": configured_host,
            "effective_host": self.server_host,
            "effective_mode": self.get_server_mode(),
            "default_public_ip": default_public_ip,
            "detected_public_ip": detected_public_ip,
        }

    def get_rpa_test_bitbrowser_id_local(self) -> str:
        return str(self._config.get("rpa", {}).get("test_bitbrowser_id_local", "") or "").strip()

    def get_rpa_test_bitbrowser_id_server(self) -> str:
        return str(self._config.get("rpa", {}).get("test_bitbrowser_id_server", "") or "").strip()

    def get_active_rpa_test_bitbrowser_id(self) -> str:
        """Get RPA test bitbrowser id based on current server mode."""
        if self.get_server_mode() == "local":
            return self.get_rpa_test_bitbrowser_id_local()
        return self.get_rpa_test_bitbrowser_id_server()

    def update_server_config(self, host: str, port: int, default_public_ip: Optional[str] = None):
        """Persist [server].host/port/default_public_ip into setting.toml, then reload memory config."""
        host_value = str(host or "").strip()
        if not host_value:
            raise ValueError("server host 不能为空")

        try:
            port_value = int(port)
        except Exception:
            raise ValueError("server port 必须是整数")

        if port_value < 1 or port_value > 65535:
            raise ValueError("server port 必须在 1-65535 之间")

        default_public_ip_value = str(
            self.default_server_public_ip if default_public_ip is None else default_public_ip
        ).strip()
        if default_public_ip_value and not self._normalize_ip_text(default_public_ip_value):
            raise ValueError("默认服务器公网IP格式无效")

        content = self._config_path.read_text(encoding="utf-8")
        content = self._upsert_toml_key_in_section(
            content=content,
            section="server",
            key="host",
            value_literal=f"\"{host_value}\"",
        )
        content = self._upsert_toml_key_in_section(
            content=content,
            section="server",
            key="port",
            value_literal=str(port_value),
        )
        content = self._upsert_toml_key_in_section(
            content=content,
            section="server",
            key="default_public_ip",
            value_literal=f"\"{default_public_ip_value}\"",
        )
        self._config_path.write_text(content, encoding="utf-8")
        self.reload_config()

    def update_rpa_test_bitbrowser_ids(self, *, local_id: str, server_id: str):
        """Persist [rpa] test bitbrowser id values into setting.toml."""
        local_value = str(local_id or "").strip()
        server_value = str(server_id or "").strip()

        content = self._config_path.read_text(encoding="utf-8")
        content = self._upsert_toml_key_in_section(
            content=content,
            section="rpa",
            key="test_bitbrowser_id_local",
            value_literal=f"\"{local_value}\"",
        )
        content = self._upsert_toml_key_in_section(
            content=content,
            section="rpa",
            key="test_bitbrowser_id_server",
            value_literal=f"\"{server_value}\"",
        )
        self._config_path.write_text(content, encoding="utf-8")
        self.reload_config()


# Global config instance
config = Config()
