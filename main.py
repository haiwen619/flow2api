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

    uvicorn.run(
        "src.main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False
    )
