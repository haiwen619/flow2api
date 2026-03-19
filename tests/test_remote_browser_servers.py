from src.core.models import (
    CaptchaConfig,
    normalize_remote_browser_servers,
    sort_remote_browser_servers_by_success,
)


def test_normalize_remote_browser_servers_from_legacy_fields():
    servers = normalize_remote_browser_servers(
        None,
        legacy_base_url="https://rb-1.example.com",
        legacy_api_key="token-1",
        legacy_timeout=75,
    )

    assert len(servers) == 1
    assert servers[0]["base_url"] == "https://rb-1.example.com"
    assert servers[0]["api_key"] == "token-1"
    assert servers[0]["timeout"] == 75


def test_sort_remote_browser_servers_by_success_count_desc():
    servers = sort_remote_browser_servers_by_success(
        [
            {"id": "a", "name": "A", "base_url": "https://a.example.com", "api_key": "a", "timeout": 60, "success_count": 1, "failure_count": 0},
            {"id": "b", "name": "B", "base_url": "https://b.example.com", "api_key": "b", "timeout": 60, "success_count": 5, "failure_count": 2},
            {"id": "c", "name": "C", "base_url": "https://c.example.com", "api_key": "c", "timeout": 60, "success_count": 5, "failure_count": 1},
        ]
    )

    assert [item["id"] for item in servers] == ["c", "b", "a"]


def test_captcha_config_syncs_primary_remote_browser_fields():
    cfg = CaptchaConfig(
        captcha_method="remote_browser",
        remote_browser_servers=[
            {
                "id": "primary",
                "name": "主节点",
                "base_url": "https://primary.example.com",
                "api_key": "primary-key",
                "timeout": 88,
                "success_count": 3,
            }
        ],
    )

    assert cfg.remote_browser_base_url == "https://primary.example.com"
    assert cfg.remote_browser_api_key == "primary-key"
    assert cfg.remote_browser_timeout == 88
