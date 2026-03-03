"""FastAPI application initialization"""
import asyncio
import json
import sys
import warnings

from fastapi import Depends, FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from .core.config import config
from .core.database import Database
from .services.flow_client import FlowClient
from .services.proxy_manager import ProxyManager
from .services.token_manager import TokenManager
from .services.load_balancer import LoadBalancer
from .services.concurrency_manager import ConcurrencyManager
from .services.generation_handler import GenerationHandler
from .api import routes, admin
from .api.accountpool import (
    AccountPoolRepository,
    AccountPoolService,
    create_accountpool_router,
)
from .api.accountpool.auth import verify_panel_token
from CommonFramePackage.proxy_pool import (
    ProxyPoolRepository,
    ProxyPoolService,
    create_proxy_pool_router,
)

REPO_ROOT = Path(__file__).parent.parent
TMP_DIR = REPO_ROOT / "tmp"
TMP_TOKEN_DIR = TMP_DIR / "Token"
STATIC_DIR = REPO_ROOT / "static"


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


async def _resolve_credential_email(credential_name: str, mode: str) -> Optional[str]:
    if str(mode or "").strip().lower() != "antigravity":
        return None
    safe_name = Path(str(credential_name or "")).name
    if not safe_name:
        return None
    file_path = TMP_TOKEN_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        return None
    return _extract_email_from_token_file(file_path)

# Playwright on Windows needs subprocess support (Proactor loop).
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# curl_cffi on Proactor emits known compatibility warnings on Windows.
# Keep Proactor for Playwright subprocess support, and suppress noisy non-fatal warnings.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"curl_cffi\.aio",
)
warnings.filterwarnings(
    "ignore",
    message=r"^Curlm alread closed! quitting from process_data$",
    category=UserWarning,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("=" * 60)
    print("Flow2API Starting...")
    print("=" * 60)

    # Get config from setting.toml
    config_dict = config.get_raw_config()

    # Check if database exists (determine if first startup)
    is_first_startup = not db.db_exists()

    # Initialize database tables structure
    await db.init_db()
    await accountpool_service.initialize()
    await proxy_pool_service.initialize()

    # Handle database initialization based on startup type
    if is_first_startup:
        print("🎉 First startup detected. Initializing database and configuration from setting.toml...")
        await db.init_config_from_toml(config_dict, is_first_startup=True)
        print("✓ Database and configuration initialized successfully.")
    else:
        print("🔄 Existing database detected. Checking for missing tables and columns...")
        await db.check_and_migrate_db(config_dict)
        print("✓ Database migration check completed.")

    # Load admin config from database
    admin_config = await db.get_admin_config()
    if admin_config:
        config.set_admin_username_from_db(admin_config.username)
        config.set_admin_password_from_db(admin_config.password)
        config.api_key = admin_config.api_key

    # Load cache configuration from database
    cache_config = await db.get_cache_config()
    config.set_cache_enabled(cache_config.cache_enabled)
    config.set_cache_timeout(cache_config.cache_timeout)
    config.set_cache_base_url(cache_config.cache_base_url or "")

    # Load generation configuration from database
    generation_config = await db.get_generation_config()
    config.set_image_timeout(generation_config.image_timeout)
    config.set_video_timeout(generation_config.video_timeout)

    # Load debug configuration from database
    debug_config = await db.get_debug_config()
    config.set_debug_enabled(debug_config.enabled)

    # Load captcha configuration from database
    captcha_config = await db.get_captcha_config()
    
    config.set_captcha_method(captcha_config.captcha_method)
    config.set_yescaptcha_api_key(captcha_config.yescaptcha_api_key)
    config.set_yescaptcha_base_url(captcha_config.yescaptcha_base_url)
    config.set_capmonster_api_key(captcha_config.capmonster_api_key)
    config.set_capmonster_base_url(captcha_config.capmonster_base_url)
    config.set_ezcaptcha_api_key(captcha_config.ezcaptcha_api_key)
    config.set_ezcaptcha_base_url(captcha_config.ezcaptcha_base_url)
    config.set_capsolver_api_key(captcha_config.capsolver_api_key)
    config.set_capsolver_base_url(captcha_config.capsolver_base_url)

    # Initialize browser captcha service if needed
    browser_service = None
    if captcha_config.captcha_method == "personal":
        from .services.browser_captcha_personal import BrowserCaptchaService
        browser_service = await BrowserCaptchaService.get_instance(db)
        print("✓ Browser captcha service initialized (nodriver mode)")
        
        # 启动常驻模式：从第一个可用token获取project_id
        tokens = await token_manager.get_all_tokens()
        resident_project_id = None
        for t in tokens:
            if t.current_project_id and t.is_active:
                resident_project_id = t.current_project_id
                break
        
        if resident_project_id:
            # 直接启动常驻模式（会自动导航到项目页面，cookie已持久化）
            await browser_service.start_resident_mode(resident_project_id)
            print(f"✓ Browser captcha resident mode started (project: {resident_project_id[:8]}...)")
        else:
            # 没有可用的project_id时，打开登录窗口供用户手动操作
            await browser_service.open_login_window()
            print("⚠ No active token with project_id found, opened login window for manual setup")
    elif captcha_config.captcha_method == "browser":
        from .services.browser_captcha import BrowserCaptchaService
        browser_service = await BrowserCaptchaService.get_instance(db)
        print("✓ Browser captcha service initialized (headless mode)")

    # Initialize concurrency manager
    tokens = await token_manager.get_all_tokens()

    await concurrency_manager.initialize(tokens)

    # Start file cache cleanup task
    await generation_handler.file_cache.start_cleanup_task()

    # Start 429 auto-unban task
    import asyncio
    async def auto_unban_task():
        """定时任务：每小时检查并解禁429被禁用的token"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时执行一次
                await token_manager.auto_unban_429_tokens()
            except Exception as e:
                print(f"❌ Auto-unban task error: {e}")

    async def auto_refresh_at_task():
        """定时任务：每分钟巡检活跃Token并触发AT自动刷新"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次
                checked, attempted, failed = await token_manager.auto_refresh_active_tokens()
                if attempted > 0 or failed > 0:
                    print(
                        f"✓ AT auto-refresh scan: checked={checked}, "
                        f"attempted={attempted}, failed={failed}"
                    )
            except Exception as e:
                print(f"❌ Auto-refresh task error: {e}")

    auto_unban_task_handle = asyncio.create_task(auto_unban_task())
    auto_refresh_task_handle = asyncio.create_task(auto_refresh_at_task())

    print(f"✓ Database initialized")
    print("✓ AccountPool initialized")
    print("✓ ProxyPool initialized")
    print(f"✓ Total tokens: {len(tokens)}")
    print(f"✓ Cache: {'Enabled' if config.cache_enabled else 'Disabled'} (timeout: {config.cache_timeout}s)")
    print(f"✓ File cache cleanup task started")
    print(f"✓ 429 auto-unban task started (runs every hour)")
    print(f"✓ AT auto-refresh task started (runs every minute)")
    print(f"✓ Server running on http://{config.server_host}:{config.server_port}")
    print("=" * 60)

    yield

    # Shutdown
    print("Flow2API Shutting down...")
    # Stop file cache cleanup task
    await generation_handler.file_cache.stop_cleanup_task()
    # Stop auto-unban task
    auto_unban_task_handle.cancel()
    try:
        await auto_unban_task_handle
    except asyncio.CancelledError:
        pass
    # Stop auto-refresh task
    auto_refresh_task_handle.cancel()
    try:
        await auto_refresh_task_handle
    except asyncio.CancelledError:
        pass
    # Close browser if initialized
    if browser_service:
        await browser_service.close()
        print("✓ Browser captcha service closed")
    print("✓ File cache cleanup task stopped")
    print("✓ 429 auto-unban task stopped")
    print("✓ AT auto-refresh task stopped")


# Initialize components
db = Database()
proxy_manager = ProxyManager(db)
flow_client = FlowClient(proxy_manager, db)
token_manager = TokenManager(db, flow_client)
concurrency_manager = ConcurrencyManager()
load_balancer = LoadBalancer(token_manager, concurrency_manager)
generation_handler = GenerationHandler(
    flow_client,
    token_manager,
    load_balancer,
    db,
    concurrency_manager,
    proxy_manager  # 添加 proxy_manager 参数
)
accountpool_repo = AccountPoolRepository()
accountpool_service = AccountPoolService(accountpool_repo)
proxy_pool_repo = ProxyPoolRepository(credential_email_resolver=_resolve_credential_email)
proxy_pool_service = ProxyPoolService(proxy_pool_repo)
proxy_manager.set_proxy_pool_service(proxy_pool_service)

# Set dependencies
routes.set_generation_handler(generation_handler)
admin.set_dependencies(token_manager, proxy_manager, db)

# Create FastAPI app
app = FastAPI(
    title="Flow2API",
    description="OpenAI-compatible API for Google VideoFX (Veo)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router)
app.include_router(admin.router)
app.include_router(create_accountpool_router(accountpool_service), tags=["AccountPool"])
app.include_router(
    create_proxy_pool_router(proxy_pool_service, verify_token=verify_panel_token),
    tags=["ProxyPool"],
)

# Static files - serve tmp directory for cached files
TMP_DIR.mkdir(exist_ok=True)
app.mount("/tmp", StaticFiles(directory=str(TMP_DIR)), name="tmp")

# HTML routes for frontend
vendor_path = STATIC_DIR / "vendor"
if vendor_path.exists():
    app.mount("/vendor", StaticFiles(directory=str(vendor_path)), name="vendor")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Redirect to login page"""
    login_file = STATIC_DIR / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    return HTMLResponse(content="<h1>Flow2API</h1><p>Frontend not found</p>", status_code=404)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Login page"""
    login_file = STATIC_DIR / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    return HTMLResponse(content="<h1>Login Page Not Found</h1>", status_code=404)


@app.get("/manage", response_class=HTMLResponse)
async def manage_page():
    """Management console page"""
    manage_file = STATIC_DIR / "manage.html"
    if manage_file.exists():
        return FileResponse(str(manage_file))
    return HTMLResponse(content="<h1>Management Page Not Found</h1>", status_code=404)


@app.get("/account_pool_page_v2_full", response_class=HTMLResponse)
async def account_pool_page():
    """Account pool automation page"""
    account_pool_file = STATIC_DIR / "account_pool_page_v2_full.html"
    if account_pool_file.exists():
        return FileResponse(str(account_pool_file))
    return HTMLResponse(content="<h1>Account Pool Page Not Found</h1>", status_code=404)


@app.get("/proxy_pool_page", response_class=HTMLResponse)
async def proxy_pool_page():
    """Proxy pool page"""
    proxy_pool_file = STATIC_DIR / "ProxyPool" / "proxy_pool_page.html"
    if proxy_pool_file.exists():
        return FileResponse(str(proxy_pool_file))
    return HTMLResponse(content="<h1>Proxy Pool Page Not Found</h1>", status_code=404)


@app.get("/creds/status")
async def creds_status(
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
    status_filter: str = Query("all"),
    mode: str = Query("antigravity"),
    token: str = Depends(verify_panel_token),
):
    """Compatibility endpoint for proxy pool credential selector."""
    _ = token
    if str(mode or "").strip().lower() != "antigravity":
        return JSONResponse(content={"success": True, "total": 0, "offset": offset, "limit": limit, "items": []})

    files = []
    if TMP_TOKEN_DIR.exists() and TMP_TOKEN_DIR.is_dir():
        files = sorted(
            TMP_TOKEN_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    token_rows = await db.get_all_tokens()
    email_to_active: Dict[str, bool] = {}
    for row in token_rows:
        email = str(getattr(row, "email", "") or "").strip().lower()
        if not email:
            continue
        email_to_active[email] = email_to_active.get(email, False) or bool(getattr(row, "is_active", False))

    normalized_filter = str(status_filter or "all").strip().lower()
    items = []
    for file_path in files:
        email = _extract_email_from_token_file(file_path) or ""
        active = email_to_active.get(email.lower()) if email else None
        disabled = active is False

        if normalized_filter in {"enabled", "active"} and disabled:
            continue
        if normalized_filter in {"disabled", "inactive"} and not disabled:
            continue

        items.append(
            {
                "filename": file_path.name,
                "user_email": email,
                "disabled": disabled,
            }
        )

    total = len(items)
    paged = items[offset: offset + limit]
    return JSONResponse(
        content={
            "success": True,
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": paged,
        }
    )
