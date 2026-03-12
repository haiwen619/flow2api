"""API routes - OpenAI compatible endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Tuple, Set
import base64
import re
import json
import time
from urllib.parse import urlparse
from curl_cffi.requests import AsyncSession
from ..core.auth import verify_api_key_header
from ..core.config import config
from ..core.models import ChatCompletionRequest
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..core.logger import debug_logger

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None


def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler


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


@router.post(
    "/v1/chat/completions",
    tags=["OpenAI API"],
    summary="统一生成接口",
    description=(
        "主生成入口，统一处理图片生成与视频生成请求。"
        "接口兼容 OpenAI Chat Completions 风格，支持 `messages` 多轮格式，也兼容部分 Gemini 风格的 `contents` 输入。"
        "\n\n主要能力："
        "\n- 文生图、图生图"
        "\n- 文生视频、图生视频、首尾帧视频、多图参考视频"
        "\n- `generationConfig.imageConfig` 自动映射图片比例与分辨率"
        "\n- `stream=true` 时返回流式结果，`stream=false` 时返回聚合结果"
        "\n\n请求头参数："
        "\n- `Authorization`：必填。格式为 `Bearer <API_KEY>`，用于主 API 鉴权。"
        "\n\n请求体参数："
        "\n- `model`：必填。指定要调用的模型。"
        "\n  - 图片模型会进入图片生成链路。"
        "\n  - 视频模型会进入视频生成链路。"
        "\n  - 对于基础图片 family，例如 `gemini-3.1-flash-image`，系统会结合 `generationConfig.imageConfig` 自动映射到具体比例/尺寸模型。"
        "\n- `messages`：推荐使用。OpenAI 风格输入。"
        "\n  - 通常传一个或多个消息对象，每个对象包含 `role` 和 `content`。"
        "\n  - `content` 可以是纯文本字符串，用于文生图或文生视频。"
        "\n  - `content` 也可以是多模态数组，常见元素为："
        "\n    - `{\"type\":\"text\",\"text\":\"你的提示词\"}`"
        "\n    - `{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,...\"}}`"
        "\n    - `{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://...\"}}`"
        "\n  - 最后一条消息会被作为当前请求的主要输入内容。"
        "\n- `contents`：兼容字段。Gemini 风格输入。"
        "\n  - 仅在调用方未使用 `messages` 时建议使用。"
        "\n  - 结构通常为 `[{\"parts\":[{\"text\":\"提示词\"}]}]`。"
        "\n  - 如果同时传了 `messages` 和 `contents`，实现上会优先走 `messages` 主链路。"
        "\n- `generationConfig`：可选。用于补充生成控制参数。"
        "\n  - `responseModalities`：声明返回模态，图片场景常见值为 `[\"IMAGE\"]`。"
        "\n  - `imageConfig`：图片模型专用配置。"
        "\n    - `aspectRatio`：图片比例。支持 `16:9`、`9:16`、`1:1`、`4:3`、`3:4` 以及等价写法如 `16x9`、`landscape`、`portrait`、`square`、`four-three`、`three-four`。"
        "\n    - `imageSize`：图片尺寸档位。支持 `1K`、`2K`、`4K` 以及等价写法如 `1024`、`2048`、`4096`。"
        "\n    - 基础图片 family 在收到这些参数后，会自动映射到具体模型后缀；不支持的比例或尺寸会返回 400。"
        "\n- `stream`：可选，默认 `false`。"
        "\n  - `true`：返回流式响应，适合前端实时接收进度或分片结果。"
        "\n  - `false`：返回一次性聚合结果，适合脚本或简单调用。"
        "\n- `image`：已废弃。历史字段，旧版调用可传 base64 图片字符串，但新调用方式应优先使用 `messages[].content[]` 中的 `image_url`。"
        "\n\n使用建议："
        "\n- 文生图/视频：`messages` 中只放文本。"
        "\n- 图生图/图生视频：`messages` 中同时放文本和一张或多张 `image_url`。"
        "\n- 首尾帧视频：在同一条消息里传文本 + 两张参考图。"
        "\n- 需要精确图片比例或分辨率时，优先使用基础图片 family + `generationConfig.imageConfig`。"
        "\n\n常见错误："
        "\n- API Key 无效会返回 401。"
        "\n- 模型不存在、比例不支持、尺寸不支持会返回 400。"
        "\n- 上游生成失败或内部处理异常通常返回 500。"
    ),
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header)
):
    """
    Flow2API 最主要的统一生成入口。

    该接口负责接收调用方请求，并根据 `model` 自动分流到图片或视频生成链路。
    它同时兼容 OpenAI 风格的 `messages` 输入，以及项目内部保留的 `contents` 输入形式。

    典型用途:
    - 文本生成图片
    - 单图或多图参考生成图片
    - 文本生成视频
    - 单图生成视频
    - 首尾帧生成视频
    - 多参考图生成视频

    关键行为:
    - 使用 Bearer API Key 鉴权
    - 自动解析 `messages` 中的文本与 `image_url`
    - 自动处理 `generationConfig.imageConfig` 的比例和尺寸映射
    - 根据 `stream` 决定返回流式 SSE 还是非流式聚合结果

    错误场景:
    - API Key 无效会返回 401
    - 模型不存在或比例/尺寸不支持会返回 400
    - 上游生成失败或内部处理异常会返回 500
    """
    try:
        # 入参摘要日志（用于排查前端/调用方参数是否被正确接收）
        debug_logger.log_info(
            f"[CHAT_COMPLETIONS][INPUT] {json.dumps(_build_request_summary_for_log(request), ensure_ascii=False)}"
        )

        # Handle both OpenAI style (messages) and Gemini style (contents)
        prompt = ""
        images: List[bytes] = []
        history_messages = request.messages or []

        if request.messages:
            last_message = request.messages[-1]
            content = last_message.content

            # OpenAI multimodal format
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
                                image_base64 = match.group(1)
                                image_bytes = base64.b64decode(image_base64)
                                images.append(image_bytes)
                        elif image_url.startswith("http://") or image_url.startswith("https://"):
                            debug_logger.log_info(f"[IMAGE_URL] 下载远程图片: {image_url}")
                            try:
                                downloaded_bytes = await retrieve_image_data(image_url)
                                if downloaded_bytes and len(downloaded_bytes) > 0:
                                    images.append(downloaded_bytes)
                                    debug_logger.log_info(f"[IMAGE_URL] ✅ 远程图片下载成功: {len(downloaded_bytes)} 字节")
                                else:
                                    debug_logger.log_warning(f"[IMAGE_URL] ⚠️ 远程图片下载失败或为空: {image_url}")
                            except Exception as e:
                                debug_logger.log_error(f"[IMAGE_URL] ❌ 远程图片下载异常: {str(e)}")

                prompt = "\n".join(prompt_parts).strip()

        elif request.contents:
            # Gemini contents format
            prompt_parts: List[str] = []
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

                    # Gemini inline image
                    inline_data = part.get("inlineData") or part.get("inline_data")
                    if isinstance(inline_data, dict):
                        image_base64 = inline_data.get("data")
                        if image_base64:
                            try:
                                images.append(base64.b64decode(image_base64))
                            except Exception as e:
                                debug_logger.log_warning(f"[INLINE_IMAGE] base64解析失败: {str(e)}")

                    # Gemini remote image
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
                                    debug_logger.log_info(f"[FILE_IMAGE] ✅ 远程图片下载成功: {len(downloaded_bytes)} 字节")
                                else:
                                    debug_logger.log_warning(f"[FILE_IMAGE] ⚠️ 远程图片下载失败或为空: {image_url}")
                            except Exception as e:
                                debug_logger.log_error(f"[FILE_IMAGE] ❌ 远程图片下载异常: {str(e)}")

            prompt = "\n".join(prompt_parts).strip()

        else:
            raise HTTPException(status_code=400, detail="Messages or contents cannot be empty")

        # Fallback to deprecated image parameter
        if request.image and not images:
            if request.image.startswith("data:image"):
                match = re.search(r"base64,(.+)", request.image)
                if match:
                    image_base64 = match.group(1)
                    image_bytes = base64.b64decode(image_base64)
                    images.append(image_bytes)

        model = request.model
        original_model = model
        auto_defaulted_model = False
        if not model and request.contents:
            generation_config = request.generationConfig if isinstance(request.generationConfig, dict) else {}
            response_modalities = generation_config.get("responseModalities", [])
            if isinstance(response_modalities, list) and any(str(modality).upper() == "IMAGE" for modality in response_modalities):
                model = "gemini-3.0-pro-image"
                auto_defaulted_model = True

        if not model:
            raise HTTPException(status_code=400, detail="Model is required")

        # 先兼容模型别名（如 -4k-16x9 / -4x3）
        alias_normalized_model = _normalize_image_model_alias(model)
        if _is_temporarily_blocked_image_model(alias_normalized_model):
            raise HTTPException(
                status_code=400,
                detail=f"暂不支持该模型: {model}",
            )

        # 根据 imageConfig 自动映射模型（如 gemini-3.0-pro-image + 4:3 -> gemini-3.0-pro-image-four-three）
        resolved_model = _resolve_model_from_image_config(alias_normalized_model, request.generationConfig)
        model = resolved_model

        # 自动参考图：仅对图片模型生效
        model_config = MODEL_CONFIG.get(model)

        generation_config_for_log = request.generationConfig if isinstance(request.generationConfig, dict) else {}
        image_config_for_log = generation_config_for_log.get("imageConfig") if isinstance(generation_config_for_log, dict) else None
        debug_logger.log_info(
            "[CHAT_COMPLETIONS][MODEL_ROUTING] "
            + json.dumps(
                _truncate_for_log(
                    {
                        "original_model": original_model,
                        "auto_defaulted_model": auto_defaulted_model,
                        "defaulted_to": model if auto_defaulted_model else None,
                        "after_alias_normalize": alias_normalized_model,
                        "imageConfig": image_config_for_log,
                        "resolved_model": model,
                        "resolved_model_config": model_config,
                    }
                ),
                ensure_ascii=False
            )
        )

        if model_config and model_config["type"] == "image" and len(history_messages) > 1:
            debug_logger.log_info(f"[CONTEXT] 开始查找历史参考图，消息数量: {len(history_messages)}")

            # 查找上一次 assistant 回复的图片
            for msg in reversed(history_messages[:-1]):
                if msg.role == "assistant" and isinstance(msg.content, str):
                    # 匹配 Markdown 图片格式: ![...](http...)
                    matches = re.findall(r"!\[.*?\]\((.*?)\)", msg.content)
                    if matches:
                        last_image_url = matches[-1]

                        if last_image_url.startswith("http"):
                            try:
                                downloaded_bytes = await retrieve_image_data(last_image_url)
                                if downloaded_bytes and len(downloaded_bytes) > 0:
                                    # 将历史图片插入到最前面
                                    images.insert(0, downloaded_bytes)
                                    debug_logger.log_info(f"[CONTEXT] ✅ 添加历史参考图: {last_image_url}")
                                    break
                                else:
                                    debug_logger.log_warning(f"[CONTEXT] 图片下载失败或为空，尝试下一个: {last_image_url}")
                            except Exception as e:
                                debug_logger.log_error(f"[CONTEXT] 处理参考图时出错: {str(e)}")
                                # 继续尝试下一个图片

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
                        "model": model,
                    }
                ),
                ensure_ascii=False
            )
        )

        # Call generation handler
        if request.stream:
            # Streaming response
            async def generate():
                async for chunk in generation_handler.handle_generation(
                    model=model,
                    prompt=prompt,
                    images=images if images else None,
                    stream=True
                ):
                    yield chunk

                # Send [DONE] signal
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            result = None
            async for chunk in generation_handler.handle_generation(
                model=model,
                prompt=prompt,
                images=images if images else None,
                stream=False
            ):
                result = chunk

            if result:
                # Parse the result JSON string
                try:
                    result_json = json.loads(result)
                    return JSONResponse(content=result_json)
                except json.JSONDecodeError:
                    # If not JSON, return as-is
                    return JSONResponse(content={"result": result})
            else:
                raise HTTPException(status_code=500, detail="Generation failed: No response from handler")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
