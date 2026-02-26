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
    return mapping.get(ratio)


def _normalize_image_model_alias(model: str) -> str:
    """
    兼容模型别名:
    - gemini-3.0-pro-image-4x3
    - gemini-3.0-pro-image-4k-16x9
    - gemini-3.0-pro-image-2k-9x16
    - gemini-3.0-pro-image-1k (等价基础分辨率)
    """
    model_normalized = str(model).strip().lower()
    if not model_normalized:
        return model

    if model_normalized in MODEL_CONFIG:
        return model_normalized

    size_suffix = None
    family_candidate = model_normalized
    for suffix in ("-4k", "-2k", "-1k"):
        if family_candidate.endswith(suffix):
            size_suffix = suffix[1:]
            family_candidate = family_candidate[: -len(suffix)]
            break

    ratio_suffix = None
    for ratio_alias in (
        "-four-three", "-three-four",
        "-landscape", "-portrait", "-square",
        "-21x9", "-16x9", "-9x16", "-1x1", "-4x3", "-3x4"
    ):
        if family_candidate.endswith(ratio_alias):
            ratio_suffix = ratio_alias[1:]
            family_candidate = family_candidate[: -len(ratio_alias)]
            break

    if family_candidate not in IMAGE_MODEL_FAMILIES:
        return model_normalized

    if ratio_suffix == "21x9":
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


@router.get("/v1/models")
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


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header)
):
    """Create chat completion (unified endpoint for image and video generation)"""
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
