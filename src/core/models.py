"""Data models for Flow2API"""
import json
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Union, Any
from datetime import datetime


SUPPORTED_CAPTCHA_METHODS_ORDER = [
    "remote_browser",
    "yescaptcha",
    "capmonster",
    "ezcaptcha",
    "capsolver",
    "browser",
    "personal",
]


def normalize_captcha_priority_order(value: Any) -> List[str]:
    """规范化验证码打码优先级顺序，仅保留显式启用的方法。"""
    parsed: List[str] = []

    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    value = decoded
                else:
                    value = [item.strip() for item in text.split(",") if item.strip()]
            except Exception:
                value = [item.strip() for item in text.split(",") if item.strip()]
        else:
            value = []

    if isinstance(value, list):
        for item in value:
            method = str(item or "").strip().lower()
            if method in SUPPORTED_CAPTCHA_METHODS_ORDER and method not in parsed:
                parsed.append(method)

    return parsed or ["remote_browser"]


class Token(BaseModel):
    """Token model for Flow2API"""
    id: Optional[int] = None

    # 认证信息 (核心)
    st: str  # Session Token (__Secure-next-auth.session-token)
    cookie: Optional[str] = None  # 完整 Cookie Header（用于 reAuth）
    cookie_file: Optional[str] = None  # Google 域名下的 Cookie Header（step4 使用）
    at: Optional[str] = None  # Access Token (从ST转换而来)
    at_expires: Optional[datetime] = None  # AT过期时间
    last_refresh_at: Optional[datetime] = None  # 最近刷新时间
    last_refresh_method: Optional[str] = None  # 最近刷新方式
    last_refresh_status: Optional[str] = None  # 最近刷新状态
    last_refresh_detail: Optional[str] = None  # 最近刷新详情

    # 基础信息
    email: str
    name: Optional[str] = ""
    remark: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    use_count: int = 0

    # VideoFX特有字段
    credits: int = 0  # 剩余credits
    user_paygate_tier: Optional[str] = None  # PAYGATE_TIER_ONE

    # 项目管理
    current_project_id: Optional[str] = None  # 当前使用的项目UUID
    current_project_name: Optional[str] = None  # 项目名称

    # 功能开关
    image_enabled: bool = True
    video_enabled: bool = True

    # 并发限制
    image_concurrency: int = -1  # -1表示无限制
    video_concurrency: int = -1  # -1表示无限制

    # 打码代理（token 级，可覆盖全局浏览器打码代理）
    captcha_proxy_url: Optional[str] = None

    # Token 禁用/封禁相关
    ban_reason: Optional[str] = None  # 禁用原因编码，如 429_rate_limit / permission_denied / google_account_disabled / manual_disabled 等
    banned_at: Optional[datetime] = None  # 禁用时间

    @model_validator(mode="before")
    @classmethod
    def _normalize_nullable_legacy_fields(cls, data):
        """兼容历史数据：数据库旧行里部分字段可能为NULL。"""
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        defaults = {
            "is_active": True,
            "use_count": 0,
            "credits": 0,
            "image_enabled": True,
            "video_enabled": True,
            "image_concurrency": -1,
            "video_concurrency": -1,
        }
        for field, default_value in defaults.items():
            if normalized.get(field) is None:
                normalized[field] = default_value
        return normalized


class Project(BaseModel):
    """Project model for VideoFX"""
    id: Optional[int] = None
    project_id: str  # VideoFX项目UUID
    token_id: int  # 关联的Token ID
    project_name: str  # 项目名称
    tool_name: str = "PINHOLE"  # 工具名称,固定为PINHOLE
    is_active: bool = True
    created_at: Optional[datetime] = None


class TokenStats(BaseModel):
    """Token statistics"""
    token_id: int
    image_count: int = 0
    video_count: int = 0
    success_count: int = 0
    error_count: int = 0  # Historical total errors (never reset)
    last_success_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    # 今日统计
    today_image_count: int = 0
    today_video_count: int = 0
    today_error_count: int = 0
    today_date: Optional[str] = None
    # 连续错误计数 (仅用于统计/排障展示)
    consecutive_error_count: int = 0


class Task(BaseModel):
    """Generation task"""
    id: Optional[int] = None
    task_id: str  # Flow API返回的operation name
    token_id: int
    model: str
    prompt: str
    status: str  # processing, completed, failed
    progress: int = 0  # 0-100
    result_urls: Optional[List[str]] = None
    error_message: Optional[str] = None
    scene_id: Optional[str] = None  # Flow API的sceneId
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RequestLog(BaseModel):
    """API request log"""
    id: Optional[int] = None
    token_id: Optional[int] = None
    operation: str
    proxy_source: Optional[str] = None
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code: int
    duration: float
    status_text: Optional[str] = None
    progress: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AdminConfig(BaseModel):
    """Admin configuration"""
    id: int = 1
    username: str
    password: str
    api_key: str
    error_ban_threshold: int = 3  # 兼容保留字段，普通请求失败已不再自动禁用 token


class ProxyConfig(BaseModel):
    """Proxy configuration"""
    id: int = 1
    enabled: bool = False  # 请求代理开关
    proxy_url: Optional[str] = None  # 请求代理地址
    media_proxy_enabled: bool = False  # 图片上传/下载代理开关
    media_proxy_url: Optional[str] = None  # 图片上传/下载代理地址


class GenerationConfig(BaseModel):
    """Generation timeout configuration"""
    id: int = 1
    image_timeout: int = 300  # seconds
    image_total_timeout: int = 120  # seconds
    video_timeout: int = 1500  # seconds


class CacheConfig(BaseModel):
    """Cache configuration"""
    id: int = 1
    cache_enabled: bool = False
    cache_timeout: int = 7200  # seconds (2 hours), 0 means never expire
    cache_base_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DebugConfig(BaseModel):
    """Debug configuration"""
    id: int = 1
    enabled: bool = False
    log_requests: bool = True
    log_responses: bool = True
    mask_token: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CaptchaConfig(BaseModel):
    """Captcha configuration"""
    id: int = 1
    captcha_method: str = "remote_browser"  # 兼容字段，实际执行以 captcha_priority_order 为准
    captcha_priority_order: List[str] = Field(default_factory=lambda: normalize_captcha_priority_order(None))
    yescaptcha_api_key: str = ""
    yescaptcha_base_url: str = "https://api.yescaptcha.com"
    capmonster_api_key: str = ""
    capmonster_base_url: str = "https://api.capmonster.cloud"
    ezcaptcha_api_key: str = ""
    ezcaptcha_base_url: str = "https://api.ez-captcha.com"
    capsolver_api_key: str = ""
    capsolver_base_url: str = "https://api.capsolver.com"
    remote_browser_base_url: str = ""
    remote_browser_api_key: str = ""
    remote_browser_timeout: int = 60
    remote_browser_proxy_enabled: bool = False  # 远程有头打码是否允许使用系统代理/代理池
    website_key: str = "6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV"
    page_action: str = "IMAGE_GENERATION"
    browser_proxy_enabled: bool = False  # 浏览器打码是否启用代理
    browser_proxy_url: Optional[str] = None  # 浏览器打码代理URL
    browser_count: int = 1  # 浏览器打码实例数量
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_priority_fields(cls, data):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        method = str(normalized.get("captcha_method") or "").strip().lower()
        order = normalize_captcha_priority_order(normalized.get("captcha_priority_order"))

        # 兼容旧版本：历史上“优先级列表”会默认包含全部方法，实际只想表达首选方式。
        if order == SUPPORTED_CAPTCHA_METHODS_ORDER:
            order = [method] if method in SUPPORTED_CAPTCHA_METHODS_ORDER else ["remote_browser"]
        elif method in SUPPORTED_CAPTCHA_METHODS_ORDER and method in order:
            order = [method] + [item for item in order if item != method]
        elif method in SUPPORTED_CAPTCHA_METHODS_ORDER and not order:
            order = [method]

        normalized["captcha_priority_order"] = order
        normalized["captcha_method"] = order[0]
        return normalized


class PluginConfig(BaseModel):
    """Plugin connection configuration"""
    id: int = 1
    connection_token: str = ""  # 插件连接token
    auto_enable_on_update: bool = True  # 更新token时自动启用（默认开启）
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# OpenAI Compatible Request Models
class ChatMessage(BaseModel):
    """Chat message"""
    role: str
    content: Union[str, List[dict]]  # string or multimodal array


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI compatible)"""
    model: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    # Gemini style compatible fields
    contents: Optional[List[dict]] = None
    generationConfig: Optional[dict] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Flow2API specific parameters
    image: Optional[str] = None  # Base64 encoded image (deprecated, use messages)
    video: Optional[str] = None  # Base64 encoded video (deprecated)
