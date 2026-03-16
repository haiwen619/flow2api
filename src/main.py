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
from .core.docker_headed_runtime import prepare_local_headed_runtime, stop_managed_xvfb
from .services.flow_client import FlowClient
from .services.proxy_manager import ProxyManager
from .services.token_manager import TokenManager
from .services.load_balancer import LoadBalancer
from .services.cluster_manager import ClusterManager
from .services.concurrency_manager import ConcurrencyManager
from .services.generation_handler import GenerationHandler
from .services.perf_monitor import perf_monitor
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
    print("Flow2API 正在启动...")
    print("=" * 60)

    runtime_server_info = config.bootstrap_runtime_server_mode()
    detected_public_ip = str(runtime_server_info.get("detected_public_ip") or "").strip()
    default_public_ip = str(runtime_server_info.get("default_public_ip") or "").strip()
    default_public_ips = runtime_server_info.get("default_public_ips") or []
    docker_headed_env = runtime_server_info.get("docker_headed_env") or {}
    default_public_ip_text = ",".join(
        str(item or "").strip() for item in default_public_ips if str(item or "").strip()
    ) or default_public_ip
    public_access_url = f"http://{detected_public_ip}:{config.server_port}" if detected_public_ip else ""

    if runtime_server_info.get("matched_server"):
        print(
            f"[启动] 检测到公网IP={detected_public_ip}，命中默认服务器IP={default_public_ip_text}，"
            f"已切换为服务器模式，绑定地址={config.server_host}"
        )
        if public_access_url:
            print(f"[启动] 公网访问地址={public_access_url}")
    elif detected_public_ip:
        print(
            f"[启动] 当前公网IP={detected_public_ip}，未命中默认服务器IP={default_public_ip_text or '<未配置>'}，"
            f"沿用配置监听地址={config.server_host}"
        )
    else:
        print(f"[启动] 未检测到公网IP，沿用配置监听地址={config.server_host}")

    if docker_headed_env.get("applied"):
        print(
            f"[启动] 检测到 Linux 服务器公网IP={detected_public_ip}，已自动设置 ALLOW_DOCKER_HEADED_CAPTCHA=true"
        )
    elif docker_headed_env.get("enabled") and docker_headed_env.get("reason") == "preconfigured":
        print("[启动] 已检测到预设 ALLOW_DOCKER_HEADED_CAPTCHA=true")

    if docker_headed_env.get("matched_linux_server") and not sys.platform.startswith("win"):
        print("[启动] 当前 Linux 服务器节点已强制仅使用 remote_browser 打码，不再回退到 browser/personal/API")

    headed_runtime = prepare_local_headed_runtime()
    if headed_runtime.get("display_applied"):
        print(f"[启动] Docker 有头浏览器已自动设置 DISPLAY={headed_runtime.get('display')}")
    if headed_runtime.get("xvfb_started"):
        print(f"[启动] Docker 有头浏览器已自动启动 Xvfb ({headed_runtime.get('display')})")
    elif headed_runtime.get("xvfb_already_running"):
        print(f"[启动] 检测到已有 Xvfb/显示服务可用 ({headed_runtime.get('display')})")
    elif headed_runtime.get("allow_headed") and headed_runtime.get("reason") not in {"", "not_docker", "headed_not_allowed", "windows"}:
        print(f"[启动] Docker 有头浏览器运行时未完全就绪: reason={headed_runtime.get('reason')}")

    # Get config from setting.toml
    config_dict = config.get_raw_config()
    print(f"[启动] 当前数据库后端: {config.db_backend}  # sqlite / mysql")

    # Check if database exists (determine if first startup)
    is_first_startup = not await db.db_exists()

    # Initialize database tables structure
    await db.init_db()
    await accountpool_service.initialize()
    await proxy_pool_service.initialize()

    # Handle database initialization based on startup type
    if is_first_startup:
        print("[启动] 检测到首次启动，正在根据 setting.toml 初始化数据库和配置...")
        await db.init_config_from_toml(config_dict, is_first_startup=True)
        print("[启动] 数据库和配置初始化完成。")
    else:
        print("[启动] 检测到已有数据库，正在检查缺失的表和字段...")
        await db.check_and_migrate_db(config_dict)
        print("[启动] 数据库迁移检查完成。")

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
    config.set_image_total_timeout(generation_config.image_total_timeout)
    config.set_video_timeout(generation_config.video_timeout)

    # Load debug configuration from database
    debug_config = await db.get_debug_config()
    config.set_debug_enabled(debug_config.enabled)

    # Load captcha configuration from database
    captcha_config = await db.get_captcha_config()
    
    config.set_captcha_method(captcha_config.captcha_method)
    config.set_captcha_priority_order(captcha_config.captcha_priority_order)
    config.set_yescaptcha_api_key(captcha_config.yescaptcha_api_key)
    config.set_yescaptcha_base_url(captcha_config.yescaptcha_base_url)
    config.set_capmonster_api_key(captcha_config.capmonster_api_key)
    config.set_capmonster_base_url(captcha_config.capmonster_base_url)
    config.set_ezcaptcha_api_key(captcha_config.ezcaptcha_api_key)
    config.set_ezcaptcha_base_url(captcha_config.ezcaptcha_base_url)
    config.set_capsolver_api_key(captcha_config.capsolver_api_key)
    config.set_capsolver_base_url(captcha_config.capsolver_base_url)
    config.set_remote_browser_base_url(captcha_config.remote_browser_base_url)
    config.set_remote_browser_api_key(captcha_config.remote_browser_api_key)
    config.set_remote_browser_timeout(captcha_config.remote_browser_timeout)

    # Initialize browser captcha service if needed
    browser_service = None
    if captcha_config.captcha_method == "personal":
        from .services.browser_captcha_personal import BrowserCaptchaService
        browser_service = await BrowserCaptchaService.get_instance(db)
        print("[启动] 浏览器打码服务已初始化（nodriver 模式）")
        
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
            print(f"[启动] 浏览器打码常驻模式已启动（项目: {resident_project_id[:8]}...）")
        else:
            # 没有可用的project_id时，打开登录窗口供用户手动操作
            await browser_service.open_login_window()
            print("[启动] 未找到带 project_id 的活跃 Token，已打开登录窗口供手动配置")
    elif captcha_config.captcha_method == "browser":
        from .services.browser_captcha import BrowserCaptchaService
        browser_service = await BrowserCaptchaService.get_instance(db)
        await browser_service.warmup_browser_slots()
        print("[启动] 浏览器打码服务已初始化（有头模式）")

    # Initialize concurrency manager
    tokens = await token_manager.get_all_tokens()

    await concurrency_manager.initialize(tokens)

    # Start performance monitor loop-lag tracker
    perf_monitor.start_loop_lag_monitor()

    # Start cluster coordination tasks
    await cluster_manager.start()

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
                print(f"[启动] 429 自动解禁任务异常: {e}")

    async def auto_refresh_at_task():
        """定时任务：每分钟巡检活跃Token并触发AT自动刷新"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次
                checked, attempted, failed = await token_manager.auto_refresh_active_tokens()
                if attempted > 0 or failed > 0:
                    print(
                        f"[启动] AT 自动刷新巡检: 已检查={checked}, "
                        f"尝试刷新={attempted}, 失败={failed}"
                    )
            except Exception as e:
                print(f"[启动] AT 自动刷新任务异常: {e}")

    auto_unban_task_handle = asyncio.create_task(auto_unban_task())
    auto_refresh_task_handle = asyncio.create_task(auto_refresh_at_task())

    print("[启动] 数据库已初始化")
    print("[启动] 账号池已初始化")
    print("[启动] 代理池已初始化")
    print(f"[启动] Token 总数: {len(tokens)}")
    print(f"[启动] 缓存状态: {'启用' if config.cache_enabled else '禁用'}（超时: {config.cache_timeout}s）")
    print("[启动] 文件缓存清理任务已启动")
    if config.cluster_enabled:
        print(f"[启动] 集群模式已启用: role={config.cluster_role}, node={config.cluster_node_name}")
        print(f"[启动] 集群对外地址: {config.cluster_effective_node_public_base_url or '<未解析到>'}")
        if config.cluster_role == "worker" and not config.cluster_effective_node_public_base_url:
            print("[启动] 警告: 当前为子节点，但未解析到可对外访问地址；容器/NAT 场景建议显式配置 cluster.node_public_base_url")
    print("[启动] 429 自动解禁任务已启动（每小时执行一次）")
    print("[启动] AT 自动刷新任务已启动（每分钟执行一次）")
    print(f"[启动] 服务绑定地址: http://{config.server_host}:{config.server_port}")
    if config.server_auto_detected and config.detected_public_ip:
        print(f"[启动] 公网访问地址: http://{config.detected_public_ip}:{config.server_port}")
    print("=" * 60)

    yield

    # Shutdown
    print("Flow2API 正在关闭...")
    stop_managed_xvfb()
    # Stop file cache cleanup task
    await generation_handler.file_cache.stop_cleanup_task()
    await cluster_manager.stop()
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
        print("[关闭] 浏览器打码服务已关闭")
    print("[关闭] 文件缓存清理任务已停止")
    print("[关闭] 429 自动解禁任务已停止")
    print("[关闭] AT 自动刷新任务已停止")


# Initialize components
db = Database()
proxy_manager = ProxyManager(db)
flow_client = FlowClient(proxy_manager, db)
token_manager = TokenManager(db, flow_client)
concurrency_manager = ConcurrencyManager()
load_balancer = LoadBalancer(token_manager, concurrency_manager)
cluster_manager = ClusterManager(load_balancer)
load_balancer.set_cluster_manager(cluster_manager)
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
routes.set_cluster_manager(cluster_manager)
admin.set_dependencies(token_manager, proxy_manager, db, concurrency_manager, load_balancer, cluster_manager)

# Create FastAPI app
app = FastAPI(
    title="Flow2API",
    summary="Flow / Gemini / Veo 的 OpenAI 兼容网关与管理后台",
    description=(
        "提供 OpenAI 兼容的生成接口、Token 管理后台、账号池、代理池，以及相关运维接口。"
        "\n\n鉴权说明："
        "\n- `/v1/*` 使用主 API Key"
        "\n- `/api/*` 管理后台接口多数使用管理员 session token"
        "\n- `/accountpool/*`、`/proxypool/*` 复用后台 session token"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "OpenAI API", "description": "对外的 OpenAI 兼容生成接口。"},
        {"name": "Admin", "description": "后台管理接口，包括 Token、配置、日志、系统信息等。"},
        {"name": "Admin Auth", "description": "后台登录态相关接口，使用独立的管理员 session token。"},
        {"name": "AccountPool", "description": "账号池管理与校验任务接口。"},
        {"name": "ProxyPool", "description": "代理池与代理凭据管理接口。"},
        {"name": "Integration", "description": "兼容性或辅助集成接口。"},
    ],
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
app.include_router(routes.router, tags=["OpenAI API"])
app.include_router(admin.router, tags=["Admin"])
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """Redirect to login page"""
    login_file = STATIC_DIR / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    return HTMLResponse(content="<h1>Flow2API</h1><p>Frontend not found</p>", status_code=404)


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    """Login page"""
    login_file = STATIC_DIR / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    return HTMLResponse(content="<h1>Login Page Not Found</h1>", status_code=404)


@app.get("/manage", response_class=HTMLResponse, include_in_schema=False)
async def manage_page():
    """Management console page"""
    manage_file = STATIC_DIR / "manage.html"
    if manage_file.exists():
        return FileResponse(str(manage_file))
    return HTMLResponse(content="<h1>Management Page Not Found</h1>", status_code=404)


@app.get("/account_pool_page_v2_full", response_class=HTMLResponse, include_in_schema=False)
async def account_pool_page():
    """Account pool automation page"""
    account_pool_file = STATIC_DIR / "account_pool_page_v2_full.html"
    if account_pool_file.exists():
        return FileResponse(str(account_pool_file))
    return HTMLResponse(content="<h1>Account Pool Page Not Found</h1>", status_code=404)


@app.get("/proxy_pool_page", response_class=HTMLResponse, include_in_schema=False)
async def proxy_pool_page():
    """Proxy pool page"""
    proxy_pool_file = STATIC_DIR / "ProxyPool" / "proxy_pool_page.html"
    if proxy_pool_file.exists():
        return FileResponse(str(proxy_pool_file))
    return HTMLResponse(content="<h1>Proxy Pool Page Not Found</h1>", status_code=404)


@app.get(
    "/creds/status",
    tags=["Integration"],
    summary="代理池凭据兼容查询",
    description="为代理池前端凭据选择器提供兼容格式的 Token 文件状态列表。",
)
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
