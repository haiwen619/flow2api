"""API routes - OpenAI compatible endpoints"""
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Tuple, Set
import base64
import re
import json
import time
from urllib.parse import urlparse
import httpx
from curl_cffi.requests import AsyncSession
from ..core.auth import verify_api_key_header
from ..core.config import config
from ..core.models import ChatCompletionRequest, RequestLog
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..core.logger import debug_logger

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None
cluster_manager = None


def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler


def set_cluster_manager(manager):
    """Set cluster manager instance"""
    global cluster_manager
    cluster_manager = manager


IMAGE_MODEL_PATTERN = re.compile(
    r"^(?P<family>.+)-(?P<ratio>landscape|portrait|square|four-three|three-four)(?:-(?P<size>2k|4k))?$"
)


def _split_image_model_id(model_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """拆分图片模型ID为 family/ratio/size"""
    match = IMAGE_MODEL_PATTERN.match(model_id)
    if not match:
        return None, None, None
    return match.group("family"), match.group("ratio"), match.group("size")


def _build_image_model_families() -> Set[str]:
    """从 MODEL_CONFIG 动态收集图片模型 family 前缀"""
    families: Set[str] = set()
    for model_id, model_config in MODEL_CONFIG.items():
        if model_config.get("type") != "image":
            continue
        family, _, _ = _split_image_model_id(model_id)
        if family:
            families.add(family)
    return families


IMAGE_MODEL_FAMILIES = _build_image_model_families()
BLOCKED_IMAGE_MODEL_FAMILIES = {"gemini-2.5-flash-image"}


def _truncate_for_log(value: Any, limit: int = 200) -> Any:
    """日志用字段裁剪，避免 base64/长文本污染日志"""
    if isinstance(value, dict):
        return {k: _truncate_for_log(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_for_log(v, limit) for v in value]
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}... (truncated, total {len(value)} chars)"
    return value


def _build_request_summary_for_log(request: ChatCompletionRequest) -> Dict[str, Any]:
    """构建 /v1/chat/completions 入参摘要日志"""
    generation_config = request.generationConfig if isinstance(request.generationConfig, dict) else {}
    image_config = generation_config.get("imageConfig") if isinstance(generation_config, dict) else None
    response_modalities = generation_config.get("responseModalities") if isinstance(generation_config, dict) else None

    summary: Dict[str, Any] = {
        "model": request.model,
        "stream": request.stream,
        "has_messages": bool(request.messages),
        "message_count": len(request.messages) if request.messages else 0,
        "has_contents": bool(request.contents),
        "contents_count": len(request.contents) if request.contents else 0,
        "responseModalities": response_modalities,
        "imageConfig": image_config,
        "has_deprecated_image": bool(request.image),
    }

    if request.messages:
        last_message = request.messages[-1]
        if isinstance(last_message.content, str):
            summary["last_message_preview"] = last_message.content[:120]
        elif isinstance(last_message.content, list):
            text_parts = [item.get("text", "") for item in last_message.content if isinstance(item, dict) and item.get("type") == "text"]
            summary["last_message_text_parts"] = len([t for t in text_parts if t])
            if text_parts:
                summary["last_message_preview"] = "\n".join([t for t in text_parts if t])[:120]

    if request.contents:
        first_parts_preview = []
        for content_item in request.contents[:1]:
            if isinstance(content_item, dict):
                parts = content_item.get("parts", [])
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict) and part.get("text"):
                            first_parts_preview.append(str(part.get("text")))
        if first_parts_preview:
            summary["contents_preview"] = "\n".join(first_parts_preview)[:120]

    return _truncate_for_log(summary)


def _cluster_dispatch_operation_name(generation_type: Optional[str]) -> str:
    normalized = str(generation_type or "").strip().lower()
    if normalized == "image":
        return "cluster_dispatch_image"
    if normalized == "video":
        return "cluster_dispatch_video"
    return "cluster_dispatch"


def _build_cluster_dispatch_request_payload(
    request: ChatCompletionRequest,
    decision: Dict[str, Any],
    target_node: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "request": _build_request_summary_for_log(request),
        "dispatch": {
            "decision_reason": str(decision.get("reason") or "").strip() or None,
            "dispatch_to": "remote",
            "target_node_id": str(target_node.get("node_id") or "").strip() or None,
            "target_node_name": str(target_node.get("node_name") or "").strip() or None,
            "target_node_base_url": str(target_node.get("base_url") or "").strip() or None,
            "target_node_weight": target_node.get("weight"),
            "target_node_available_slots": target_node.get("available_slots"),
            "target_node_active_requests": target_node.get("active_requests"),
        },
    }


def _parse_cluster_dispatch_response_payload(
    *,
    content_type: str,
    body_bytes: Optional[bytes] = None,
    body_text: Optional[str] = None,
) -> Any:
    text = str(body_text or "").strip()
    if body_bytes is not None and not text:
        try:
            text = body_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    if text and "json" in str(content_type or "").lower():
        try:
            return _truncate_for_log(json.loads(text), 1200)
        except Exception:
            pass

    if text:
        return _truncate_for_log(text, 1200)
    return None


async def _write_cluster_dispatch_request_log(
    *,
    request: ChatCompletionRequest,
    decision: Dict[str, Any],
    target_node: Dict[str, Any],
    status_code: int,
    duration_seconds: float,
    status_text: str,
    progress: int,
    response_payload: Dict[str, Any],
    log_id: Optional[int] = None,
) -> Optional[int]:
    if not cluster_manager or not cluster_manager.is_master():
        return log_id
    if generation_handler is None or getattr(generation_handler, "db", None) is None:
        return log_id

    request_payload = _build_cluster_dispatch_request_payload(request, decision, target_node)
    request_body = json.dumps(_truncate_for_log(request_payload, 1200), ensure_ascii=False)
    response_body = json.dumps(_truncate_for_log(response_payload, 1200), ensure_ascii=False)
    safe_progress = max(0, min(100, int(progress or 0)))

    try:
        if log_id:
            await generation_handler.db.update_request_log(
                int(log_id),
                token_id=None,
                operation=_cluster_dispatch_operation_name(
                    target_node.get("dispatch_generation_type") or decision.get("generation_type")
                ),
                proxy_source="cluster_remote",
                request_body=request_body,
                response_body=response_body,
                status_code=int(status_code),
                duration=float(duration_seconds or 0),
                status_text=str(status_text or "").strip() or "cluster_dispatched",
                progress=safe_progress,
            )
            return int(log_id)

        row = RequestLog(
            token_id=None,
            operation=_cluster_dispatch_operation_name(
                target_node.get("dispatch_generation_type") or decision.get("generation_type")
            ),
            proxy_source="cluster_remote",
            request_body=request_body,
            response_body=response_body,
            status_code=int(status_code),
            duration=float(duration_seconds or 0),
            status_text=str(status_text or "").strip() or "cluster_dispatched",
            progress=safe_progress,
        )
        return await generation_handler.db.add_request_log(row)
    except Exception as exc:
        debug_logger.log_warning(f"[CLUSTER] 写入主节点分流请求日志失败: {exc}")
        return log_id


def _is_temporarily_blocked_image_model(model_id: str) -> bool:
    """Check whether the requested image model family is currently blocked by runtime config."""
    normalized = str(model_id or "").strip().lower()
    if not normalized or not bool(config.block_gemini_25_flash_image):
        return False
    if normalized in BLOCKED_IMAGE_MODEL_FAMILIES:
        return True
    family, _, _ = _split_image_model_id(normalized)
    return bool(family and family in BLOCKED_IMAGE_MODEL_FAMILIES)


def _is_unsupported_ultrawide_ratio(value: Optional[str]) -> bool:
    """判断是否为当前不支持的 21:9 超宽屏参数"""
    if not value:
        return False
    compact = str(value).strip().lower().replace(" ", "").replace("_", "-").replace("/", ":")
    return compact in {"21:9", "21x9", "ultrawide", "ultra-wide", "cinema", "cinema-screen"}


def _build_available_family_variants(family: str) -> List[str]:
    """列出某个图片 family 下可用的全部模型"""
    return sorted(
        model_id
        for model_id, model_config in MODEL_CONFIG.items()
        if model_config.get("type") == "image" and _split_image_model_id(model_id)[0] == family
    )


def _select_fallback_image_model(
    family: str,
    target_ratio: str,
    target_size: Optional[str]
) -> Optional[str]:
    """
    当精确比例/尺寸模型不存在时，选择同 family 下最接近的可用模型。
    例如:
    - square -> landscape/portrait
    - four-three -> landscape
    - three-four -> portrait
    """
    ratio_fallback_order = {
        "square": ["square", "landscape", "portrait", "four-three", "three-four"],
        "four-three": ["four-three", "landscape", "square", "portrait", "three-four"],
        "three-four": ["three-four", "portrait", "square", "landscape", "four-three"],
        "landscape": ["landscape", "four-three", "square", "portrait", "three-four"],
        "portrait": ["portrait", "three-four", "square", "landscape", "four-three"],
    }
    ratios_to_try = ratio_fallback_order.get(target_ratio, [target_ratio, "landscape", "portrait"])

    size_candidates: List[Optional[str]] = [target_size, None] if target_size else [None]
    # 去重并保持顺序
    seen_sizes = set()
    normalized_size_candidates: List[Optional[str]] = []
    for size in size_candidates:
        if size not in seen_sizes:
            seen_sizes.add(size)
            normalized_size_candidates.append(size)

    for ratio in ratios_to_try:
        for size in normalized_size_candidates:
            suffix = f"-{size}" if size else ""
            candidate = f"{family}-{ratio}{suffix}"
            if candidate in MODEL_CONFIG:
                return candidate

    # 兜底: 返回 family 下第一个可用模型
    available = _build_available_family_variants(family)
    return available[0] if available else None


def _normalize_ratio_suffix_from_model(model_ratio: Optional[str]) -> Optional[str]:
    """模型后缀中的比例别名 -> 标准比例后缀"""
    if not model_ratio:
        return None

    ratio = model_ratio.strip().lower()
    compact = ratio.replace(" ", "").replace("_", "-").replace("/", "x").replace(":", "x")
    mapping = {
        "landscape": "landscape",
        "16x9": "landscape",
        "portrait": "portrait",
        "9x16": "portrait",
        "square": "square",
        "1x1": "square",
        "four-three": "four-three",
        "4x3": "four-three",
        "three-four": "three-four",
        "3x4": "three-four",
    }
    return mapping.get(compact) or mapping.get(ratio)


def _normalize_image_model_alias(model: str) -> str:
    """
    兼容模型别名:
    - gemini-3.0-pro-image-4x3
    - gemini-3.0-pro-image-4k-16x9
    - gemini-3.0-pro-image-2k-9x16
    - gemini-3.0-pro-image-1k (等价基础分辨率)
    - gemini-3.1-flash-image-4x3
    - gemini-3.1-flash-image-4k-16x9
    - gemini-3.1-flash-image-2k-9x16
    - gemini-3.1-flash-image-1k (等价基础分辨率)
    """
    model_normalized = str(model).strip().lower()
    if not model_normalized:
        return model

    if model_normalized in MODEL_CONFIG:
        return model_normalized

    family_candidate = None
    alias_suffix = ""
    for family in sorted(IMAGE_MODEL_FAMILIES, key=len, reverse=True):
        if model_normalized == family:
            family_candidate = family
            alias_suffix = ""
            break
        prefix = f"{family}-"
        if model_normalized.startswith(prefix):
            family_candidate = family
            alias_suffix = model_normalized[len(prefix):]
            break

    if not family_candidate:
        return model_normalized

    size_suffix = None
    ratio_suffix = None
    if alias_suffix:
        tokens = [token for token in alias_suffix.split("-") if token]
        size_tokens = [token for token in tokens if token in {"1k", "2k", "4k"}]
        if len(size_tokens) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model alias '{model}': multiple image size suffixes found."
            )

        if size_tokens:
            size_suffix = size_tokens[0]
            size_removed = False
            ratio_tokens: List[str] = []
            for token in tokens:
                if not size_removed and token == size_suffix:
                    size_removed = True
                    continue
                ratio_tokens.append(token)
            ratio_suffix = "-".join(ratio_tokens) if ratio_tokens else None
        else:
            ratio_suffix = "-".join(tokens) if tokens else None

    ratio_suffix_compact = (
        str(ratio_suffix or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "-")
        .replace("/", "x")
        .replace(":", "x")
    )
    if ratio_suffix_compact == "21x9":
        raise HTTPException(
            status_code=400,
            detail="Unsupported model ratio suffix: 21x9. 21:9 is not supported by current image models."
        )

    normalized_ratio = _normalize_ratio_suffix_from_model(ratio_suffix)
    if ratio_suffix and not normalized_ratio:
        raise HTTPException(status_code=400, detail=f"Unsupported model ratio suffix: {ratio_suffix}")

    target_ratio = normalized_ratio or "landscape"
    target_size = None if size_suffix in (None, "1k") else size_suffix
    size_part = f"-{target_size}" if target_size else ""
    candidate_model = f"{family_candidate}-{target_ratio}{size_part}"

    if candidate_model in MODEL_CONFIG:
        return candidate_model

    available_variants = _build_available_family_variants(family_candidate)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Cannot map model alias '{model}' to available model. "
            f"Available variants: {', '.join(available_variants)}"
        )
    )


def _normalize_aspect_ratio(aspect_ratio: Optional[str]) -> Optional[str]:
    """将 imageConfig.aspectRatio 统一映射到模型后缀"""
    if not aspect_ratio:
        return None

    value = str(aspect_ratio).strip().lower()
    compact = value.replace(" ", "").replace("_", "-").replace("/", ":")

    if _is_unsupported_ultrawide_ratio(compact):
        raise HTTPException(
            status_code=400,
            detail="Unsupported imageConfig.aspectRatio: 21:9. 21:9 is not supported by current image models."
        )

    mapping = {
        "16:9": "landscape",
        "16x9": "landscape",
        "landscape": "landscape",
        "image-aspect-ratio-landscape": "landscape",
        "9:16": "portrait",
        "9x16": "portrait",
        "portrait": "portrait",
        "image-aspect-ratio-portrait": "portrait",
        "1:1": "square",
        "1x1": "square",
        "square": "square",
        "image-aspect-ratio-square": "square",
        "4:3": "four-three",
        "4x3": "four-three",
        "four-three": "four-three",
        "image-aspect-ratio-landscape-four-three": "four-three",
        "3:4": "three-four",
        "3x4": "three-four",
        "three-four": "three-four",
        "image-aspect-ratio-portrait-three-four": "three-four",
    }

    return mapping.get(compact)


def _normalize_image_size(image_size: Optional[str]) -> Optional[str]:
    """将 imageConfig.imageSize 统一映射到模型后缀"""
    if image_size is None:
        return None

    value = str(image_size).strip().lower().replace(" ", "").replace("_", "")
    if not value:
        return ""

    if value in {"1k", "1024"} or "1k" in value:
        return ""
    if value in {"2k", "2048"} or "2k" in value:
        return "2k"
    if value in {"4k", "4096"} or "4k" in value:
        return "4k"

    return "__unsupported__"


def _resolve_model_from_image_config(model: str, generation_config: Optional[Dict[str, Any]]) -> str:
    """根据 generationConfig.imageConfig 自动选择具体图片模型"""
    if not generation_config or not isinstance(generation_config, dict):
        if model in IMAGE_MODEL_FAMILIES:
            default_model = f"{model}-landscape"
            if default_model in MODEL_CONFIG:
                return default_model
        return model

    image_config = generation_config.get("imageConfig")
    if not isinstance(image_config, dict):
        if model in IMAGE_MODEL_FAMILIES:
            default_model = f"{model}-landscape"
            if default_model in MODEL_CONFIG:
                return default_model
        return model

    raw_aspect_ratio = image_config.get("aspectRatio")
    raw_image_size = image_config.get("imageSize")

    normalized_aspect_ratio = _normalize_aspect_ratio(raw_aspect_ratio)
    normalized_image_size = _normalize_image_size(raw_image_size)

    if raw_aspect_ratio and not normalized_aspect_ratio:
        raise HTTPException(status_code=400, detail=f"Unsupported imageConfig.aspectRatio: {raw_aspect_ratio}")

    if raw_image_size is not None and normalized_image_size == "__unsupported__":
        raise HTTPException(status_code=400, detail=f"Unsupported imageConfig.imageSize: {raw_image_size}")

    # 没有任何可映射参数时保持原模型
    if not raw_aspect_ratio and raw_image_size is None:
        if model in IMAGE_MODEL_FAMILIES:
            default_model = f"{model}-landscape"
            if default_model in MODEL_CONFIG:
                return default_model
        return model

    requested_model_config = MODEL_CONFIG.get(model)
    family = None
    inferred_ratio = None
    inferred_size = None

    parsed_family, parsed_ratio, parsed_size = _split_image_model_id(model)

    if requested_model_config and requested_model_config.get("type") == "image":
        if parsed_family:
            family, inferred_ratio, inferred_size = parsed_family, parsed_ratio, parsed_size
        elif model in IMAGE_MODEL_FAMILIES:
            family = model
        else:
            return model
    else:
        if model in IMAGE_MODEL_FAMILIES:
            family = model
        elif parsed_family and parsed_family in IMAGE_MODEL_FAMILIES:
            family, inferred_ratio, inferred_size = parsed_family, parsed_ratio, parsed_size
        else:
            return model

    target_ratio = normalized_aspect_ratio or inferred_ratio or "landscape"

    # imageSize 未显式传入时，保留原模型 size（如 -2k/-4k）
    if raw_image_size is None:
        target_size = inferred_size
    else:
        target_size = normalized_image_size

    size_suffix = f"-{target_size}" if target_size else ""
    target_model = f"{family}-{target_ratio}{size_suffix}"

    if target_model in MODEL_CONFIG:
        return target_model

    # 精确模型不存在时，自动回退到最接近可用模型（避免 1:1 在仅支持横竖屏 family 上直接报错）
    fallback_model = _select_fallback_image_model(family, target_ratio, target_size)
    if fallback_model and fallback_model in MODEL_CONFIG:
        debug_logger.log_warning(
            f"[MODEL_ROUTING] 精确模型不存在，已自动回退: {target_model} -> {fallback_model}"
        )
        return fallback_model

    available_variants = _build_available_family_variants(family)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Cannot map model '{model}' with imageConfig(aspectRatio={raw_aspect_ratio}, imageSize={raw_image_size}). "
            f"Available variants: {', '.join(available_variants)}"
        )
    )


async def retrieve_image_data(url: str) -> Optional[bytes]:
    """
    智能获取图片数据：
    1. 优先检查是否为本地 /tmp/ 缓存文件，如果是则直接读取磁盘
    2. 如果本地不存在或是外部链接，则进行网络下载
    """
    # 优先尝试本地读取
    try:
        if "/tmp/" in url and generation_handler and generation_handler.file_cache:
            path = urlparse(url).path
            filename = path.split("/tmp/")[-1]
            local_file_path = generation_handler.file_cache.cache_dir / filename

            if local_file_path.exists() and local_file_path.is_file():
                data = local_file_path.read_bytes()
                if data:
                    return data
    except Exception as e:
        debug_logger.log_warning(f"[CONTEXT] 本地缓存读取失败: {str(e)}")

    # 回退逻辑：网络下载
    try:
        async with AsyncSession() as session:
            response = await session.get(url, timeout=30, impersonate="chrome110", verify=False)
            if response.status_code == 200:
                return response.content
            else:
                debug_logger.log_warning(f"[CONTEXT] 图片下载失败，状态码: {response.status_code}")
    except Exception as e:
        debug_logger.log_error(f"[CONTEXT] 图片下载异常: {str(e)}")

    return None


@router.get(
    "/v1/models",
    tags=["OpenAI API"],
    summary="获取可用模型列表",
    description="返回当前服务支持的图片与视频模型，兼容 OpenAI 风格模型列表格式。",
)
async def list_models():
    """List available models"""
    models = []

    for model_id, config in MODEL_CONFIG.items():
        description = f"{config['type'].capitalize()} generation"
        if config['type'] == 'image':
            description += f" - {config['model_name']}"
        else:
            description += f" - {config['model_key']}"

        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "flow2api",
            "description": description
        })

    return {
        "object": "list",
        "data": models
    }

async def verify_internal_cluster_request(
    x_cluster_key: Optional[str] = Header(None, alias="X-Cluster-Key"),
):
    """Authenticate cluster internal requests."""
    if not cluster_manager or not cluster_manager.is_enabled():
        raise HTTPException(status_code=403, detail="Cluster mode is disabled")
    if not cluster_manager.verify_cluster_key(x_cluster_key):
        raise HTTPException(status_code=401, detail="Invalid cluster key")
    return True


def _resolve_requested_model_id(request: ChatCompletionRequest) -> tuple[str, str, Optional[str], bool]:
    model = request.model
    original_model = model
    auto_defaulted_model = False
    if not model and request.contents:
        generation_config = request.generationConfig if isinstance(request.generationConfig, dict) else {}
        response_modalities = generation_config.get("responseModalities", [])
        if isinstance(response_modalities, list) and any(
            str(modality).upper() == "IMAGE" for modality in response_modalities
        ):
            model = "gemini-3.0-pro-image"
            auto_defaulted_model = True

    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    alias_normalized_model = _normalize_image_model_alias(model)
    if _is_temporarily_blocked_image_model(alias_normalized_model):
        raise HTTPException(status_code=400, detail=f"暂不支持该模型: {model}")

    resolved_model = _resolve_model_from_image_config(alias_normalized_model, request.generationConfig)
    model_config = MODEL_CONFIG.get(resolved_model)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {resolved_model}")
    return resolved_model, model_config["type"], original_model, auto_defaulted_model


def _resolve_cluster_target_base_url(target_node: Dict[str, Any]) -> str:
    base_url = str(target_node.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return ""

    try:
        parsed = urlparse(base_url)
    except Exception:
        return base_url

    if parsed.port:
        return base_url

    try:
        reported_port = int(target_node.get("server_port") or 0)
    except Exception:
        reported_port = 0

    if reported_port <= 0:
        return base_url

    if (parsed.scheme == "http" and reported_port == 80) or (parsed.scheme == "https" and reported_port == 443):
        return base_url

    hostname = str(parsed.hostname or "").strip()
    if not hostname:
        return base_url

    netloc = f"{hostname}:{reported_port}"
    return parsed._replace(netloc=netloc).geturl().rstrip("/")


async def _proxy_cluster_chat_completion(request: ChatCompletionRequest, decision: Dict[str, Any]):
    target_node = dict(decision.get("node") or {})
    base_url = _resolve_cluster_target_base_url(target_node)
    if not base_url:
        raise HTTPException(status_code=503, detail="选中的子节点缺少可访问地址")

    endpoint = f"{base_url}/api/internal/cluster/chat/completions"
    timeout_seconds = max(10, int(config.cluster_dispatch_timeout_seconds))
    payload = request.model_dump(exclude_none=True)
    headers = {
        "X-Cluster-Key": config.cluster_key,
        "X-Cluster-Node-Id": config.cluster_node_id,
        "X-Cluster-Origin-Role": config.cluster_role,
        "X-Cluster-Target-Node": str(target_node.get("node_id") or ""),
    }
    response_headers = {
        "X-Flow2API-Dispatch": "remote",
        "X-Flow2API-Node": str(target_node.get("node_name") or target_node.get("node_id") or ""),
    }
    started_at = time.perf_counter()
    target_node_id = str(target_node.get("node_id") or "").strip()
    target_node_name = str(target_node.get("node_name") or target_node_id).strip() or target_node_id
    generation_type = str(decision.get("node", {}).get("dispatch_generation_type") or "").strip() or None
    resolved_model = str(decision.get("node", {}).get("dispatch_model") or "").strip() or None
    decision["generation_type"] = generation_type
    dispatch_log_id: Optional[int] = None

    try:
        if request.stream:
            client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
            upstream_response = await client.send(
                client.build_request("POST", endpoint, headers=headers, json=payload),
                stream=True,
            )
            if upstream_response.status_code >= 400:
                body = await upstream_response.aread()
                if cluster_manager:
                    await cluster_manager.record_dispatch(
                        target_node_id=target_node_id,
                        target_node_name=target_node_name,
                        generation_type=generation_type,
                        resolved_model=resolved_model,
                        status_code=upstream_response.status_code,
                        duration_ms=int((time.perf_counter() - started_at) * 1000),
                        error=body.decode("utf-8", errors="ignore")[:500],
                    )
                await _write_cluster_dispatch_request_log(
                    request=request,
                    decision=decision,
                    target_node=target_node,
                    status_code=upstream_response.status_code,
                    duration_seconds=(time.perf_counter() - started_at),
                    status_text="cluster_dispatch_failed",
                    progress=0,
                    response_payload={
                        "dispatch": {
                            "target_node_id": target_node_id,
                            "target_node_name": target_node_name,
                            "mode": "stream",
                            "remote_status_code": upstream_response.status_code,
                        },
                        "error": _parse_cluster_dispatch_response_payload(
                            content_type=upstream_response.headers.get("content-type", "application/json"),
                            body_bytes=body,
                        ),
                    },
                )
                await upstream_response.aclose()
                await client.aclose()
                return Response(
                    content=body,
                    status_code=upstream_response.status_code,
                    media_type=upstream_response.headers.get("content-type", "application/json"),
                    headers=response_headers,
                )

            if cluster_manager:
                await cluster_manager.record_dispatch(
                    target_node_id=target_node_id,
                    target_node_name=target_node_name,
                    generation_type=generation_type,
                    resolved_model=resolved_model,
                    status_code=upstream_response.status_code,
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )

            dispatch_log_id = await _write_cluster_dispatch_request_log(
                request=request,
                decision=decision,
                target_node=target_node,
                status_code=102,
                duration_seconds=0,
                status_text="cluster_dispatch_streaming",
                progress=15,
                response_payload={
                    "dispatch": {
                        "target_node_id": target_node_id,
                        "target_node_name": target_node_name,
                        "mode": "stream",
                        "remote_status_code": upstream_response.status_code,
                    },
                    "message": "主节点已将请求分流到子节点，正在转发流式响应",
                },
            )

            async def iter_stream():
                stream_error: Optional[str] = None
                try:
                    async for chunk in upstream_response.aiter_bytes():
                        if chunk:
                            yield chunk
                except Exception as exc:
                    stream_error = str(exc)
                    raise
                finally:
                    await upstream_response.aclose()
                    await client.aclose()
                    await _write_cluster_dispatch_request_log(
                        request=request,
                        decision=decision,
                        target_node=target_node,
                        status_code=upstream_response.status_code if not stream_error else 502,
                        duration_seconds=(time.perf_counter() - started_at),
                        status_text="cluster_dispatched" if not stream_error else "cluster_dispatch_failed",
                        progress=100 if not stream_error else 0,
                        response_payload={
                            "dispatch": {
                                "target_node_id": target_node_id,
                                "target_node_name": target_node_name,
                                "mode": "stream",
                                "remote_status_code": upstream_response.status_code,
                                "stream_completed": not bool(stream_error),
                            },
                            "error": stream_error or None,
                            "message": "主节点已完成子节点流式响应转发" if not stream_error else "主节点转发子节点流式响应失败",
                        },
                        log_id=dispatch_log_id,
                    )

            stream_headers = dict(response_headers)
            stream_headers["Cache-Control"] = "no-cache"
            stream_headers["Connection"] = "keep-alive"
            stream_headers["X-Accel-Buffering"] = "no"
            return StreamingResponse(
                iter_stream(),
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get("content-type", "text/event-stream"),
                headers=stream_headers,
            )

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            upstream_response = await client.post(endpoint, headers=headers, json=payload)
        if cluster_manager:
            await cluster_manager.record_dispatch(
                target_node_id=target_node_id,
                target_node_name=target_node_name,
                generation_type=generation_type,
                resolved_model=resolved_model,
                status_code=upstream_response.status_code,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                error=None if upstream_response.status_code < 400 else upstream_response.text[:500],
            )
        await _write_cluster_dispatch_request_log(
            request=request,
            decision=decision,
            target_node=target_node,
            status_code=upstream_response.status_code,
            duration_seconds=(time.perf_counter() - started_at),
            status_text="cluster_dispatched" if upstream_response.status_code < 400 else "cluster_dispatch_failed",
            progress=100 if upstream_response.status_code < 400 else 0,
            response_payload={
                "dispatch": {
                    "target_node_id": target_node_id,
                    "target_node_name": target_node_name,
                    "mode": "json",
                    "remote_status_code": upstream_response.status_code,
                },
                "response": _parse_cluster_dispatch_response_payload(
                    content_type=upstream_response.headers.get("content-type", "application/json"),
                    body_bytes=upstream_response.content,
                ),
            },
        )
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
            headers=response_headers,
        )
    except HTTPException:
        raise
    except Exception as exc:
        if cluster_manager:
            await cluster_manager.record_dispatch(
                target_node_id=target_node_id,
                target_node_name=target_node_name,
                generation_type=generation_type,
                resolved_model=resolved_model,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                error=str(exc),
            )
        await _write_cluster_dispatch_request_log(
            request=request,
            decision=decision,
            target_node=target_node,
            status_code=502,
            duration_seconds=(time.perf_counter() - started_at),
            status_text="cluster_dispatch_failed",
            progress=0,
            response_payload={
                "dispatch": {
                    "target_node_id": target_node_id,
                    "target_node_name": target_node_name,
                    "mode": "exception",
                },
                "error": str(exc),
            },
            log_id=dispatch_log_id,
        )
        raise HTTPException(status_code=502, detail=f"主节点转发到子节点失败: {exc}") from exc


async def _execute_local_chat_completion(
    request: ChatCompletionRequest,
    *,
    resolved_model: str,
    generation_type: str,
    original_model: Optional[str],
    auto_defaulted_model: bool,
    response_headers: Optional[Dict[str, str]] = None,
):
    debug_logger.log_info(
        f"[CHAT_COMPLETIONS][INPUT] {json.dumps(_build_request_summary_for_log(request), ensure_ascii=False)}"
    )

    prompt = ""
    images: List[bytes] = []
    history_messages = request.messages or []

    if request.messages:
        last_message = request.messages[-1]
        content = last_message.content
        if isinstance(content, str):
            prompt = content
        elif isinstance(content, list):
            prompt_parts: List[str] = []
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        prompt_parts.append(text)
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        match = re.search(r"base64,(.+)", image_url)
                        if match:
                            images.append(base64.b64decode(match.group(1)))
                    elif image_url.startswith("http://") or image_url.startswith("https://"):
                        debug_logger.log_info(f"[IMAGE_URL] 下载远程图片: {image_url}")
                        try:
                            downloaded_bytes = await retrieve_image_data(image_url)
                            if downloaded_bytes and len(downloaded_bytes) > 0:
                                images.append(downloaded_bytes)
                                debug_logger.log_info(
                                    f"[IMAGE_URL] ✅ 远程图片下载成功: {len(downloaded_bytes)} 字节"
                                )
                            else:
                                debug_logger.log_warning(f"[IMAGE_URL] ⚠️ 远程图片下载失败或为空: {image_url}")
                        except Exception as exc:
                            debug_logger.log_error(f"[IMAGE_URL] ❌ 远程图片下载异常: {str(exc)}")
            prompt = "\n".join(prompt_parts).strip()
    elif request.contents:
        prompt_parts = []
        for content_item in request.contents:
            if not isinstance(content_item, dict):
                continue
            parts = content_item.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if text:
                    prompt_parts.append(text)

                inline_data = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline_data, dict):
                    image_base64 = inline_data.get("data")
                    if image_base64:
                        try:
                            images.append(base64.b64decode(image_base64))
                        except Exception as exc:
                            debug_logger.log_warning(f"[INLINE_IMAGE] base64解析失败: {str(exc)}")

                file_data = part.get("fileData") or part.get("file_data")
                if isinstance(file_data, dict):
                    image_url = file_data.get("fileUri") or file_data.get("file_uri")
                    if isinstance(image_url, str) and (
                        image_url.startswith("http://") or image_url.startswith("https://")
                    ):
                        debug_logger.log_info(f"[FILE_IMAGE] 下载远程图片: {image_url}")
                        try:
                            downloaded_bytes = await retrieve_image_data(image_url)
                            if downloaded_bytes and len(downloaded_bytes) > 0:
                                images.append(downloaded_bytes)
                                debug_logger.log_info(
                                    f"[FILE_IMAGE] ✅ 远程图片下载成功: {len(downloaded_bytes)} 字节"
                                )
                            else:
                                debug_logger.log_warning(f"[FILE_IMAGE] ⚠️ 远程图片下载失败或为空: {image_url}")
                        except Exception as exc:
                            debug_logger.log_error(f"[FILE_IMAGE] ❌ 远程图片下载异常: {str(exc)}")
        prompt = "\n".join(prompt_parts).strip()
    else:
        raise HTTPException(status_code=400, detail="Messages or contents cannot be empty")

    if request.image and not images and request.image.startswith("data:image"):
        match = re.search(r"base64,(.+)", request.image)
        if match:
            images.append(base64.b64decode(match.group(1)))

    model_config = MODEL_CONFIG.get(resolved_model)
    generation_config_for_log = request.generationConfig if isinstance(request.generationConfig, dict) else {}
    image_config_for_log = (
        generation_config_for_log.get("imageConfig") if isinstance(generation_config_for_log, dict) else None
    )
    debug_logger.log_info(
        "[CHAT_COMPLETIONS][MODEL_ROUTING] "
        + json.dumps(
            _truncate_for_log(
                {
                    "original_model": original_model,
                    "auto_defaulted_model": auto_defaulted_model,
                    "defaulted_to": resolved_model if auto_defaulted_model else None,
                    "imageConfig": image_config_for_log,
                    "resolved_model": resolved_model,
                    "resolved_model_config": model_config,
                    "generation_type": generation_type,
                }
            ),
            ensure_ascii=False,
        )
    )

    if model_config and model_config["type"] == "image" and len(history_messages) > 1:
        debug_logger.log_info(f"[CONTEXT] 开始查找历史参考图，消息数量: {len(history_messages)}")
        for msg in reversed(history_messages[:-1]):
            if msg.role != "assistant" or not isinstance(msg.content, str):
                continue
            matches = re.findall(r"!\[.*?\]\((.*?)\)", msg.content)
            if not matches:
                continue
            last_image_url = matches[-1]
            if not last_image_url.startswith("http"):
                continue
            try:
                downloaded_bytes = await retrieve_image_data(last_image_url)
                if downloaded_bytes and len(downloaded_bytes) > 0:
                    images.insert(0, downloaded_bytes)
                    debug_logger.log_info(f"[CONTEXT] ✅ 添加历史参考图: {last_image_url}")
                    break
                debug_logger.log_warning(f"[CONTEXT] 图片下载失败或为空，尝试下一个: {last_image_url}")
            except Exception as exc:
                debug_logger.log_error(f"[CONTEXT] 处理参考图时出错: {str(exc)}")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    debug_logger.log_info(
        "[CHAT_COMPLETIONS][PARSED] "
        + json.dumps(
            _truncate_for_log(
                {
                    "prompt_preview": prompt[:160],
                    "prompt_length": len(prompt),
                    "image_count": len(images),
                    "stream": request.stream,
                    "model": resolved_model,
                    "generation_type": generation_type,
                }
            ),
            ensure_ascii=False,
        )
    )

    local_slot_acquired = False
    if cluster_manager:
        local_slot_acquired = await cluster_manager.acquire_local_slot()
        if not local_slot_acquired:
            raise HTTPException(status_code=503, detail="当前节点并发已满，请稍后重试")

    if request.stream:
        async def generate():
            try:
                async for chunk in generation_handler.handle_generation(
                    model=resolved_model,
                    prompt=prompt,
                    images=images if images else None,
                    stream=True,
                ):
                    yield chunk
                yield "data: [DONE]\n\n"
            finally:
                if local_slot_acquired and cluster_manager:
                    await cluster_manager.release_local_slot()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if response_headers:
            headers.update(response_headers)
        return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)

    try:
        result = None
        async for chunk in generation_handler.handle_generation(
            model=resolved_model,
            prompt=prompt,
            images=images if images else None,
            stream=False,
        ):
            result = chunk
    finally:
        if local_slot_acquired and cluster_manager:
            await cluster_manager.release_local_slot()

    if not result:
        raise HTTPException(status_code=500, detail="Generation failed: No response from handler")

    try:
        response = JSONResponse(content=json.loads(result))
    except json.JSONDecodeError:
        response = JSONResponse(content={"result": result})
    if response_headers:
        for key, value in response_headers.items():
            response.headers[key] = value
    return response


async def _handle_chat_completion_request(
    request: ChatCompletionRequest,
    *,
    allow_cluster_dispatch: bool,
):
    try:
        resolved_model, generation_type, original_model, auto_defaulted_model = _resolve_requested_model_id(request)
        if allow_cluster_dispatch and cluster_manager and cluster_manager.should_dispatch_externally():
            decision = await cluster_manager.choose_dispatch_target(
                resolved_model=resolved_model,
                generation_type=generation_type,
            )
            if decision.get("dispatch_to") == "remote":
                return await _proxy_cluster_chat_completion(request, decision)

            response_headers = {
                "X-Flow2API-Dispatch": "local",
                "X-Flow2API-Node": config.cluster_node_name,
            }
            return await _execute_local_chat_completion(
                request,
                resolved_model=resolved_model,
                generation_type=generation_type,
                original_model=original_model,
                auto_defaulted_model=auto_defaulted_model,
                response_headers=response_headers,
            )

        return await _execute_local_chat_completion(
            request,
            resolved_model=resolved_model,
            generation_type=generation_type,
            original_model=original_model,
            auto_defaulted_model=auto_defaulted_model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/v1/chat/completions",
    tags=["OpenAI API"],
    summary="统一生成接口",
    description=(
        "主生成入口，统一处理图片生成与视频生成请求。"
        "接口兼容 OpenAI Chat Completions 风格，支持 `messages` 多轮格式，也兼容部分 Gemini 风格的 `contents` 输入。"
        "\n\n请求头参数："
        "\n- `Authorization`：必填。格式为 `Bearer <API_KEY>`，用于主 API 鉴权。"
    ),
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header),
):
    _ = api_key
    return await _handle_chat_completion_request(request, allow_cluster_dispatch=True)


@router.post("/api/internal/cluster/chat/completions", include_in_schema=False)
async def execute_cluster_chat_completion(
    request: ChatCompletionRequest,
    authorized: bool = Depends(verify_internal_cluster_request),
):
    _ = authorized
    return await _handle_chat_completion_request(request, allow_cluster_dispatch=False)
