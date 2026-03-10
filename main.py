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
    from src.main import app

    runtime_server_info = config.bootstrap_runtime_server_mode()
    detected_public_ip = str(runtime_server_info.get("detected_public_ip") or "").strip()
    default_public_ip = str(runtime_server_info.get("default_public_ip") or "").strip()
    if runtime_server_info.get("matched_server"):
        print(
            f"[startup] 检测到公网IP={detected_public_ip}，命中默认服务器IP={default_public_ip}，"
            f"已切换为服务器模式，监听地址={config.server_host}"
        )
    elif detected_public_ip:
        print(
            f"[startup] 当前公网IP={detected_public_ip}，未命中默认服务器IP={default_public_ip or '<未配置>'}，"
            f"沿用配置监听地址={config.server_host}"
        )
    else:
        print(
            f"[startup] 未检测到公网IP，沿用配置监听地址={config.server_host}"
        )

    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        reload=False
    )
