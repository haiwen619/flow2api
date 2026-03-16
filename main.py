"""Flow2API - Main Entry Point"""
import asyncio
import sys

import uvicorn

if __name__ == "__main__":
    # Playwright requires subprocess support on Windows event loop.
    # Use Proactor policy to avoid NotImplementedError in create_subprocess_exec.
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass

    from src.core.config import config
    from src.core.docker_headed_runtime import prepare_local_headed_runtime
    from src.main import app

    runtime_server_info = config.bootstrap_runtime_server_mode()
    detected_public_ip = str(runtime_server_info.get("detected_public_ip") or "").strip()
    default_public_ip = str(runtime_server_info.get("default_public_ip") or "").strip()
    default_public_ips = runtime_server_info.get("default_public_ips") or []
    docker_headed_env = runtime_server_info.get("docker_headed_env") or {}
    default_public_ip_text = ",".join(str(item or "").strip() for item in default_public_ips if str(item or "").strip()) or default_public_ip
    public_access_url = f"http://{detected_public_ip}:{config.server_port}" if detected_public_ip else ""
    if runtime_server_info.get("matched_server"):
        print(
            f"[startup] 检测到公网IP={detected_public_ip}，命中默认服务器IP={default_public_ip_text}，"
            f"已切换为服务器模式，绑定地址={config.server_host}"
        )
        if public_access_url:
            print(f"[startup] 公网访问地址={public_access_url}")
    elif detected_public_ip:
        print(
            f"[startup] 当前公网IP={detected_public_ip}，未命中默认服务器IP={default_public_ip_text or '<未配置>'}，"
            f"沿用配置监听地址={config.server_host}"
        )
    else:
        print(
            f"[startup] 未检测到公网IP，沿用配置监听地址={config.server_host}"
        )

    if docker_headed_env.get("applied"):
        print(
            f"[startup] 检测到 Linux 服务器公网IP={detected_public_ip}，已自动设置 ALLOW_DOCKER_HEADED_CAPTCHA=true"
        )
    elif docker_headed_env.get("enabled") and docker_headed_env.get("reason") == "preconfigured":
        print("[startup] 已检测到预设 ALLOW_DOCKER_HEADED_CAPTCHA=true")

    if docker_headed_env.get("matched_linux_server") and not sys.platform.startswith("win"):
        print("[startup] 当前 Linux 服务器节点已强制仅使用 remote_browser 打码，不再回退到 browser/personal/API")

    headed_runtime = prepare_local_headed_runtime()
    if headed_runtime.get("display_applied"):
        print(f"[startup] Docker 有头浏览器已自动设置 DISPLAY={headed_runtime.get('display')}")
    if headed_runtime.get("xvfb_started"):
        print(f"[startup] Docker 有头浏览器已自动启动 Xvfb ({headed_runtime.get('display')})")
    elif headed_runtime.get("xvfb_already_running"):
        print(f"[startup] 检测到已有 Xvfb/显示服务可用 ({headed_runtime.get('display')})")
    elif headed_runtime.get("allow_headed") and headed_runtime.get("reason") not in {"", "not_docker", "headed_not_allowed", "windows"}:
        print(f"[startup] Docker 有头浏览器运行时未完全就绪: reason={headed_runtime.get('reason')}")

    print(f"[startup] 当前数据库后端: {config.db_backend}  # sqlite / mysql")

    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        reload=False
    )
