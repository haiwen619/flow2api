"""后台图片并发自测服务。"""

import asyncio
import json
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import Response

from ..core.models import ChatCompletionRequest
from ..services.generation_handler import MODEL_CONFIG


class ImageLoadTestService:
    """管理后台触发的图片并发自测任务。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._job: Optional[Dict[str, Any]] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._image_models: List[str] = sorted(
            [
                str(model_id)
                for model_id, meta in MODEL_CONFIG.items()
                if isinstance(meta, dict) and str(meta.get("type") or "").strip().lower() == "image"
            ]
        )

    async def start_job(
        self,
        *,
        model: str,
        total_requests: int,
        duration_seconds: int,
        max_concurrency: int,
        timeout_seconds: int,
        prompt_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            if self._job and self._job.get("status") in {"queued", "running", "stopping"}:
                raise ValueError("当前已有图片并发自测任务在运行，请先等待完成或停止")

            now = datetime.now().isoformat(timespec="seconds")
            job_id = uuid.uuid4().hex[:12]
            prompt_prefix = str(prompt_prefix or "").strip()
            interval_seconds = duration_seconds / max(total_requests, 1)
            self._job = {
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "model": str(model or "random").strip() or "random",
                "total_requests": int(total_requests),
                "duration_seconds": int(duration_seconds),
                "max_concurrency": int(max_concurrency),
                "timeout_seconds": int(timeout_seconds),
                "prompt_prefix": prompt_prefix,
                "target_rps": round(total_requests / max(duration_seconds, 1), 2),
                "launch_interval_ms": int(interval_seconds * 1000),
                "launched": 0,
                "completed": 0,
                "succeeded": 0,
                "failed": 0,
                "in_flight": 0,
                "cancel_requested": False,
                "recent_errors": [],
                "recent_prompts": [],
                "duration_samples_ms": [],
                "last_error": None,
                "summary": "任务已创建，等待启动",
            }
            self._runner_task = asyncio.create_task(self._run_job(job_id))
            return self._build_snapshot_locked()

    async def stop_job(self) -> Dict[str, Any]:
        async with self._lock:
            if not self._job:
                return {"running": False, "job": None}
            if self._job.get("status") in {"completed", "failed", "cancelled"}:
                return {"running": False, "job": self._build_snapshot_locked()}
            self._job["cancel_requested"] = True
            if self._job.get("status") in {"queued", "running"}:
                self._job["status"] = "stopping"
                self._job["summary"] = "已请求停止，等待已发起请求完成"
            return {"running": True, "job": self._build_snapshot_locked()}

    async def get_status(self) -> Dict[str, Any]:
        async with self._lock:
            if not self._job:
                return {"running": False, "job": None}
            status = str(self._job.get("status") or "")
            return {
                "running": status in {"queued", "running", "stopping"},
                "job": self._build_snapshot_locked(),
            }

    async def _run_job(self, job_id: str) -> None:
        semaphore: Optional[asyncio.Semaphore] = None
        request_tasks: List[asyncio.Task] = []
        try:
            async with self._lock:
                if not self._job or self._job.get("job_id") != job_id:
                    return
                self._job["status"] = "running"
                self._job["started_at"] = datetime.now().isoformat(timespec="seconds")
                self._job["summary"] = "任务运行中，开始按节奏发起请求"
                total_requests = int(self._job.get("total_requests") or 0)
                duration_seconds = int(self._job.get("duration_seconds") or 0)
                max_concurrency = int(self._job.get("max_concurrency") or 1)
                model = str(self._job.get("model") or "")
                timeout_seconds = int(self._job.get("timeout_seconds") or 180)
                prompt_prefix = str(self._job.get("prompt_prefix") or "")

            semaphore = asyncio.Semaphore(max(1, max_concurrency))
            start_mono = time.perf_counter()
            interval_seconds = duration_seconds / max(total_requests, 1)

            for index in range(total_requests):
                if await self._is_cancel_requested(job_id):
                    break
                launch_at = start_mono + index * interval_seconds
                sleep_seconds = launch_at - time.perf_counter()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                if await self._is_cancel_requested(job_id):
                    break

                prompt = self._build_prompt(index=index, prompt_prefix=prompt_prefix)
                await self._record_launch(job_id, prompt)
                request_tasks.append(
                    asyncio.create_task(
                        self._run_single_request(
                            job_id=job_id,
                            model=model,
                            prompt=prompt,
                            timeout_seconds=timeout_seconds,
                            semaphore=semaphore,
                        )
                    )
                )

            if request_tasks:
                await asyncio.gather(*request_tasks, return_exceptions=True)

            async with self._lock:
                if not self._job or self._job.get("job_id") != job_id:
                    return
                cancelled = bool(self._job.get("cancel_requested"))
                self._job["status"] = "cancelled" if cancelled else "completed"
                self._job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                if cancelled:
                    self._job["summary"] = "任务已停止，未继续发起剩余请求"
                else:
                    self._job["summary"] = "任务已完成，全部请求已结束"
        except Exception as exc:
            async with self._lock:
                if self._job and self._job.get("job_id") == job_id:
                    self._job["status"] = "failed"
                    self._job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                    self._job["last_error"] = str(exc)
                    self._job["summary"] = f"任务执行失败：{str(exc)}"
                    self._push_recent_error_locked(f"任务异常：{str(exc)}")
        finally:
            async with self._lock:
                if self._job and self._job.get("job_id") == job_id:
                    self._job["in_flight"] = max(0, int(self._job.get("in_flight") or 0))

    async def _run_single_request(
        self,
        *,
        job_id: str,
        model: str,
        prompt: str,
        timeout_seconds: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            started_at = time.perf_counter()
            await self._increase_inflight(job_id)
            success = False
            error_message: Optional[str] = None

            try:
                from ..api.routes import _handle_chat_completion_request

                request_model = self._choose_request_model(model)
                request = ChatCompletionRequest(
                    model=request_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                response = await asyncio.wait_for(
                    _handle_chat_completion_request(request, allow_cluster_dispatch=True),
                    timeout=max(10, int(timeout_seconds)),
                )
                status_code, body_text = self._extract_response_payload(response)
                success = status_code < 400
                if not success:
                    error_message = body_text or f"HTTP {status_code}"
            except HTTPException as exc:
                error_message = str(exc.detail or exc)
            except asyncio.TimeoutError:
                error_message = f"请求超时（>{int(timeout_seconds)} 秒）"
            except Exception as exc:
                error_message = str(exc)
            finally:
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                await self._complete_request(
                    job_id=job_id,
                    success=success,
                    duration_ms=duration_ms,
                    error_message=error_message,
                )

    async def _record_launch(self, job_id: str, prompt: str) -> None:
        async with self._lock:
            if not self._job or self._job.get("job_id") != job_id:
                return
            self._job["launched"] = int(self._job.get("launched") or 0) + 1
            prompts = self._job.setdefault("recent_prompts", [])
            prompts.insert(0, prompt)
            del prompts[8:]
            launched = int(self._job.get("launched") or 0)
            total = int(self._job.get("total_requests") or 0)
            self._job["summary"] = f"已发起 {launched}/{total} 个请求"

    async def _increase_inflight(self, job_id: str) -> None:
        async with self._lock:
            if not self._job or self._job.get("job_id") != job_id:
                return
            self._job["in_flight"] = int(self._job.get("in_flight") or 0) + 1

    async def _complete_request(
        self,
        *,
        job_id: str,
        success: bool,
        duration_ms: int,
        error_message: Optional[str],
    ) -> None:
        async with self._lock:
            if not self._job or self._job.get("job_id") != job_id:
                return

            self._job["in_flight"] = max(0, int(self._job.get("in_flight") or 0) - 1)
            self._job["completed"] = int(self._job.get("completed") or 0) + 1
            if success:
                self._job["succeeded"] = int(self._job.get("succeeded") or 0) + 1
            else:
                self._job["failed"] = int(self._job.get("failed") or 0) + 1
                self._job["last_error"] = error_message
                self._push_recent_error_locked(error_message or "未知错误")

            samples = self._job.setdefault("duration_samples_ms", [])
            samples.append(int(duration_ms))
            if len(samples) > 400:
                del samples[:-400]

            completed = int(self._job.get("completed") or 0)
            total = int(self._job.get("total_requests") or 0)
            self._job["summary"] = f"已完成 {completed}/{total} 个请求"

    async def _is_cancel_requested(self, job_id: str) -> bool:
        async with self._lock:
            return bool(self._job and self._job.get("job_id") == job_id and self._job.get("cancel_requested"))

    def _build_snapshot_locked(self) -> Dict[str, Any]:
        job = dict(self._job or {})
        if not job:
            return {}

        samples = [int(item) for item in job.get("duration_samples_ms") or [] if isinstance(item, (int, float))]
        started_at = job.get("started_at")
        finished_at = job.get("finished_at")
        elapsed_seconds = self._calculate_elapsed_seconds(started_at, finished_at)
        completed = int(job.get("completed") or 0)
        launched = int(job.get("launched") or 0)
        total = max(1, int(job.get("total_requests") or 0))
        succeeded = int(job.get("succeeded") or 0)
        failed = int(job.get("failed") or 0)

        return {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "model": job.get("model"),
            "total_requests": total,
            "duration_seconds": int(job.get("duration_seconds") or 0),
            "max_concurrency": int(job.get("max_concurrency") or 0),
            "timeout_seconds": int(job.get("timeout_seconds") or 0),
            "prompt_prefix": job.get("prompt_prefix") or "",
            "target_rps": job.get("target_rps") or 0,
            "launch_interval_ms": int(job.get("launch_interval_ms") or 0),
            "launched": launched,
            "completed": completed,
            "succeeded": succeeded,
            "failed": failed,
            "in_flight": int(job.get("in_flight") or 0),
            "cancel_requested": bool(job.get("cancel_requested")),
            "launch_progress_percent": round(launched / total * 100, 1),
            "complete_progress_percent": round(completed / total * 100, 1),
            "success_rate": round((succeeded / completed) * 100, 1) if completed else 0.0,
            "avg_duration_ms": round(sum(samples) / len(samples), 1) if samples else 0.0,
            "p95_duration_ms": self._percentile(samples, 95),
            "max_duration_ms": max(samples) if samples else 0,
            "recent_errors": list(job.get("recent_errors") or []),
            "recent_prompts": list(job.get("recent_prompts") or []),
            "last_error": job.get("last_error"),
            "summary": job.get("summary") or "",
        }

    def _choose_request_model(self, configured_model: str) -> str:
        normalized = str(configured_model or "").strip()
        if normalized and normalized.lower() not in {"random", "随机"}:
            return normalized
        if not self._image_models:
            raise HTTPException(status_code=500, detail="没有可用的图片模型用于并发自测")
        return random.choice(self._image_models)

    def _push_recent_error_locked(self, message: str) -> None:
        errors = self._job.setdefault("recent_errors", []) if self._job else []
        errors.insert(0, str(message or "未知错误"))
        del errors[8:]

    @staticmethod
    def _extract_response_payload(response: Any) -> (int, str):
        if isinstance(response, Response):
            body = getattr(response, "body", b"") or b""
            if isinstance(body, bytes):
                text = body.decode("utf-8", errors="ignore")
            else:
                text = str(body)
            return int(getattr(response, "status_code", 200) or 200), text[:500]

        if isinstance(response, dict):
            return 200, json.dumps(response, ensure_ascii=False)[:500]

        return 200, str(response)[:500]

    @staticmethod
    def _calculate_elapsed_seconds(started_at: Optional[str], finished_at: Optional[str]) -> int:
        if not started_at:
            return 0
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(finished_at) if finished_at else datetime.now()
            return max(0, int((end_dt - start_dt).total_seconds()))
        except Exception:
            return 0

    @staticmethod
    def _percentile(samples: List[int], percentile: int) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        if len(ordered) == 1:
            return float(ordered[0])
        index = max(0, min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))))
        return float(ordered[index])

    @staticmethod
    def _build_prompt(*, index: int, prompt_prefix: str) -> str:
        subjects = [
            "一只橘猫宇航员",
            "赛博朋克风格的城市街景",
            "雨后霓虹灯下的机车少女",
            "雪山脚下的玻璃小屋",
            "蒸汽朋克机械猫头鹰",
            "海边日落时的灯塔",
            "古风庭院里的白狐",
            "未来感无人机港口",
            "森林中的萤火虫鹿群",
            "月球基地外的探测车",
        ]
        styles = [
            "电影级光影",
            "超写实摄影",
            "高饱和插画",
            "柔和水彩",
            "精致 3D 渲染",
            "国风数字艺术",
            "低多边形艺术",
            "梦幻概念设定图",
        ]
        scenes = [
            "主角位于画面中央，背景层次丰富",
            "广角构图，强调空间纵深",
            "近景特写，突出细节质感",
            "清晨薄雾环境，带体积光",
            "夜景霓虹反射，地面略有积水",
            "加入轻微动态模糊和飞散粒子",
            "画面干净，适合做壁纸",
            "强调高对比和清晰主体",
        ]
        details = [
            "色彩统一，细节丰富，禁止水印和文字",
            "画质清晰，构图完整，避免多余肢体",
            "适合商业海报，突出主体辨识度",
            "纹理真实，边缘干净，光线自然",
            "高细节，高质量，层次分明",
        ]
        base_prompt = (
            f"{random.choice(subjects)}，{random.choice(styles)}，{random.choice(scenes)}，"
            f"{random.choice(details)}。第 {index + 1} 次并发自测样本。"
        )
        prefix = str(prompt_prefix or "").strip()
        return f"{prefix}，{base_prompt}" if prefix else base_prompt


image_load_test_service = ImageLoadTestService()