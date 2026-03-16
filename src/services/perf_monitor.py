"""Lightweight in-memory performance monitor for high-concurrency diagnostics.

Collects rolling metrics with minimal overhead. All data is ephemeral
(lost on restart) — this is intentional to avoid adding DB I/O pressure.
"""

import asyncio
import importlib
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class RequestRecord:
    """One completed/active request snapshot."""
    request_id: str
    started_at: float
    finished_at: Optional[float] = None
    duration_ms: Optional[int] = None
    operation: str = ""  # generate_image / generate_video
    token_id: Optional[int] = None
    status: str = "processing"  # processing / success / failed
    # Breakdown timings (ms)
    token_select_ms: int = 0
    ensure_at_ms: int = 0
    ensure_project_ms: int = 0
    generation_pipeline_ms: int = 0
    slot_wait_ms: int = 0
    current_phase: str = "started"
    waiting_on_image_slot: bool = False


class PerfMonitor:
    """Process-wide performance metrics collector."""

    def __init__(self, history_size: int = 500, window_seconds: int = 300):
        """
        Args:
            history_size: Max number of completed requests to keep.
            window_seconds: Sliding window for rate calculations.
        """
        self._history_size = history_size
        self._window_seconds = window_seconds
        # Completed requests ring-buffer
        self._completed: Deque[RequestRecord] = deque(maxlen=history_size)
        # Currently in-flight requests
        self._inflight: Dict[str, RequestRecord] = {}
        self._lock = asyncio.Lock()
        # Counters since process start
        self._total_requests = 0
        self._total_success = 0
        self._total_failed = 0
        self._process_start = time.time()
        # Event loop lag tracking
        self._last_loop_check = time.monotonic()
        self._loop_lag_samples: Deque[float] = deque(maxlen=60)  # 1 per second for 60s
        self._loop_lag_task: Optional[asyncio.Task] = None
        # Network throughput tracking (system-wide)
        self._network_samples: Deque[Dict[str, float]] = deque(maxlen=60)
        self._network_task: Optional[asyncio.Task] = None
        self._last_net_bytes_sent: Optional[int] = None
        self._last_net_bytes_recv: Optional[int] = None
        self._last_net_sample_at: Optional[float] = None
        self._loadtest_events: Deque[Dict[str, Any]] = deque(maxlen=600)
        self._logical_cpu_count = max(1, os.cpu_count() or 1)
        self._cpu_psutil = None
        self._cpu_process = None
        self._cpu_counter_primed = False
        self._prime_cpu_counter()
        _ResourceInfo.prime_system_cpu_counter()

    def _normalize_process_cpu_percent(self, value: float) -> float:
        """Normalize process CPU usage to Task Manager semantics."""
        try:
            cpu = float(value)
        except Exception:
            return -1.0
        if cpu < 0:
            return -1.0
        return round(max(0.0, cpu) / self._logical_cpu_count, 1)

    def _prime_cpu_counter(self) -> None:
        """Prime psutil cpu_percent so subsequent reads are meaningful."""
        try:
            self._cpu_psutil = importlib.import_module("psutil")
            self._cpu_process = self._cpu_psutil.Process(os.getpid())
            self._cpu_process.cpu_percent(interval=None)
            self._cpu_counter_primed = True
        except Exception:
            self._cpu_psutil = None
            self._cpu_process = None
            self._cpu_counter_primed = False

    def start_loop_lag_monitor(self):
        """Start a background task that measures event-loop lag every second."""
        if self._loop_lag_task is None or self._loop_lag_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._loop_lag_task = loop.create_task(self._measure_loop_lag())
                if self._network_task is None or self._network_task.done():
                    self._network_task = loop.create_task(self._measure_network_io())
            except RuntimeError:
                pass

    async def _measure_loop_lag(self):
        """Periodically schedule a callback and measure how late it fires."""
        while True:
            before = time.monotonic()
            await asyncio.sleep(0.1)  # target 100ms
            after = time.monotonic()
            lag_ms = max(0, (after - before - 0.1) * 1000)
            self._loop_lag_samples.append(round(lag_ms, 1))
            await asyncio.sleep(0.9)  # total ~1s per sample

    async def _measure_network_io(self):
        """Sample system-wide network throughput once per second."""
        while True:
            sample = _ResourceInfo.get_network_counters()
            if sample:
                now = time.monotonic()
                sent = int(sample.get("bytes_sent", 0) or 0)
                recv = int(sample.get("bytes_recv", 0) or 0)
                if (
                    self._last_net_bytes_sent is not None
                    and self._last_net_bytes_recv is not None
                    and self._last_net_sample_at is not None
                ):
                    elapsed = max(0.001, now - self._last_net_sample_at)
                    send_bps = max(0.0, (sent - self._last_net_bytes_sent) / elapsed)
                    recv_bps = max(0.0, (recv - self._last_net_bytes_recv) / elapsed)
                    self._network_samples.append({
                        "send_kbps": round(send_bps / 1024, 1),
                        "recv_kbps": round(recv_bps / 1024, 1),
                    })
                self._last_net_bytes_sent = sent
                self._last_net_bytes_recv = recv
                self._last_net_sample_at = now
            await asyncio.sleep(1.0)

    # ---- Request lifecycle ----

    async def on_request_start(self, request_id: str, operation: str = "") -> None:
        async with self._lock:
            rec = RequestRecord(
                request_id=request_id,
                started_at=time.time(),
                operation=operation,
            )
            self._inflight[request_id] = rec
            self._total_requests += 1

    async def on_request_progress(
        self,
        request_id: str,
        *,
        token_id: Optional[int] = None,
        current_phase: Optional[str] = None,
        waiting_on_image_slot: Optional[bool] = None,
    ) -> None:
        """Update an in-flight request with lightweight real-time progress metadata."""
        async with self._lock:
            rec = self._inflight.get(request_id)
            if rec is None:
                return
            if token_id is not None:
                rec.token_id = token_id
            if current_phase is not None:
                rec.current_phase = str(current_phase or "").strip() or rec.current_phase
            if waiting_on_image_slot is not None:
                rec.waiting_on_image_slot = bool(waiting_on_image_slot)

    async def on_request_end(
        self,
        request_id: str,
        success: bool,
        token_id: Optional[int] = None,
        perf_trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with self._lock:
            rec = self._inflight.pop(request_id, None)
            if rec is None:
                # Not tracked (maybe started before monitor was ready)
                return
            now = time.time()
            rec.finished_at = now
            rec.duration_ms = int((now - rec.started_at) * 1000)
            rec.status = "success" if success else "failed"
            rec.token_id = token_id
            if perf_trace:
                rec.token_select_ms = perf_trace.get("token_select_ms", 0)
                rec.ensure_at_ms = perf_trace.get("ensure_at_ms", 0)
                rec.ensure_project_ms = perf_trace.get("ensure_project_ms", 0)
                rec.generation_pipeline_ms = perf_trace.get("generation_pipeline_ms", 0)
                rec.slot_wait_ms = perf_trace.get("slot_wait_ms", 0)
            rec.waiting_on_image_slot = False
            if success:
                self._total_success += 1
            else:
                self._total_failed += 1
            self._completed.append(rec)

    # ---- Snapshot for dashboard ----

    async def get_diagnostics(self) -> Dict[str, Any]:
        """Return a comprehensive diagnostics snapshot."""
        async with self._lock:
            now = time.time()
            uptime_s = int(now - self._process_start)

            # Current in-flight
            inflight_list = []
            for rec in self._inflight.values():
                elapsed_ms = int((now - rec.started_at) * 1000)
                inflight_list.append({
                    "request_id": rec.request_id,
                    "operation": rec.operation,
                    "token_id": rec.token_id,
                    "elapsed_ms": elapsed_ms,
                    "current_phase": rec.current_phase,
                    "waiting_on_image_slot": rec.waiting_on_image_slot,
                })
            inflight_list.sort(key=lambda x: -x["elapsed_ms"])

            image_slot_waiting = [item for item in inflight_list if item.get("waiting_on_image_slot")]

            # Sliding window stats
            cutoff = now - self._window_seconds
            window_records = [r for r in self._completed if r.finished_at and r.finished_at >= cutoff]

            window_count = len(window_records)
            window_success = sum(1 for r in window_records if r.status == "success")
            window_failed = sum(1 for r in window_records if r.status == "failed")

            # RPM (requests per minute) calculation
            if window_count > 0 and self._window_seconds > 0:
                rpm = round(window_count / (self._window_seconds / 60), 1)
            else:
                rpm = 0

            # Duration percentiles
            durations = sorted([r.duration_ms for r in window_records if r.duration_ms is not None])
            duration_stats = self._calc_percentiles(durations)

            # Breakdown averages
            token_select_times = [r.token_select_ms for r in window_records if r.token_select_ms > 0]
            ensure_at_times = [r.ensure_at_ms for r in window_records if r.ensure_at_ms > 0]
            ensure_project_times = [r.ensure_project_ms for r in window_records if r.ensure_project_ms > 0]
            pipeline_times = [r.generation_pipeline_ms for r in window_records if r.generation_pipeline_ms > 0]
            slot_wait_times = [r.slot_wait_ms for r in window_records if r.slot_wait_ms > 0]

            # Per-token distribution in window
            token_usage: Dict[int, int] = {}
            for r in window_records:
                if r.token_id is not None:
                    token_usage[r.token_id] = token_usage.get(r.token_id, 0) + 1

            # Recent slow requests (top 10 by duration)
            slow_requests = sorted(
                [r for r in window_records if r.duration_ms is not None],
                key=lambda r: -r.duration_ms
            )[:10]
            slow_list = [{
                "request_id": r.request_id,
                "operation": r.operation,
                "token_id": r.token_id,
                "duration_ms": r.duration_ms,
                "token_select_ms": r.token_select_ms,
                "ensure_at_ms": r.ensure_at_ms,
                "ensure_project_ms": r.ensure_project_ms,
                "generation_pipeline_ms": r.generation_pipeline_ms,
                "slot_wait_ms": r.slot_wait_ms,
                "status": r.status,
            } for r in slow_requests]

            # Event loop lag
            lag_samples = list(self._loop_lag_samples)
            avg_lag = round(sum(lag_samples) / len(lag_samples), 1) if lag_samples else 0
            max_lag = round(max(lag_samples), 1) if lag_samples else 0

            # System resources
            sys_info = _ResourceInfo.get_info()

            # Network throughput
            net_samples = list(self._network_samples)
            latest_net = net_samples[-1] if net_samples else {"send_kbps": 0, "recv_kbps": 0}
            avg_send = round(sum(x.get("send_kbps", 0) for x in net_samples) / len(net_samples), 1) if net_samples else 0
            avg_recv = round(sum(x.get("recv_kbps", 0) for x in net_samples) / len(net_samples), 1) if net_samples else 0
            max_send = round(max((x.get("send_kbps", 0) for x in net_samples), default=0), 1)
            max_recv = round(max((x.get("recv_kbps", 0) for x in net_samples), default=0), 1)

            # Throughput timeline (per-minute buckets for last 5 minutes)
            timeline = self._build_timeline(window_records, now, bucket_seconds=60, buckets=5)

            return {
                "uptime_seconds": uptime_s,
                "inflight": {
                    "count": len(inflight_list),
                    "requests": inflight_list[:20],  # cap display
                },
                "queues": {
                    "image_slot_waiting_count": len(image_slot_waiting),
                    "image_slot_waiting_requests": image_slot_waiting[:20],
                },
                "totals": {
                    "requests": self._total_requests,
                    "success": self._total_success,
                    "failed": self._total_failed,
                },
                "window": {
                    "seconds": self._window_seconds,
                    "count": window_count,
                    "success": window_success,
                    "failed": window_failed,
                    "rpm": rpm,
                    "success_rate": round(window_success / window_count * 100, 1) if window_count > 0 else 0,
                },
                "duration": duration_stats,
                "breakdown": {
                    "token_select_ms": self._calc_percentiles(sorted(token_select_times)),
                    "ensure_at_ms": self._calc_percentiles(sorted(ensure_at_times)),
                    "ensure_project_ms": self._calc_percentiles(sorted(ensure_project_times)),
                    "generation_pipeline_ms": self._calc_percentiles(sorted(pipeline_times)),
                    "slot_wait_ms": self._calc_percentiles(sorted(slot_wait_times)),
                },
                "token_distribution": token_usage,
                "slow_requests": slow_list,
                "event_loop": {
                    "avg_lag_ms": avg_lag,
                    "max_lag_ms": max_lag,
                    "lag_samples": lag_samples[-30:],  # last 30s
                },
                "network": {
                    "latest_send_kbps": latest_net.get("send_kbps", 0),
                    "latest_recv_kbps": latest_net.get("recv_kbps", 0),
                    "avg_send_kbps": avg_send,
                    "avg_recv_kbps": avg_recv,
                    "max_send_kbps": max_send,
                    "max_recv_kbps": max_recv,
                    "samples": net_samples[-30:],
                },
                "system": sys_info,
                "timeline": timeline,
            }

    def get_realtime_cpu_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight system-wide CPU snapshot for on-demand preview."""
        return _ResourceInfo.get_system_snapshot()

    def get_runtime_snapshot(self, include_top_processes: bool = False, top_process_limit: int = 5) -> Dict[str, Any]:
        """Return a lightweight combined runtime snapshot for ad-hoc diagnostics."""
        snapshot = {
            "sampled_at": time.time(),
            "process": _ResourceInfo.get_info(),
            "system": _ResourceInfo.get_system_snapshot(),
        }
        if include_top_processes:
            snapshot["top_processes"] = _ResourceInfo.get_top_processes(limit=top_process_limit)
        return snapshot

    def record_loadtest_event(self, source: str, payload: Dict[str, Any]) -> None:
        """Append one image load-test diagnostic event for dashboard analysis."""
        try:
            event_payload = dict(payload or {})
            event_payload["source"] = str(source or "unknown")
            event_payload["recorded_at"] = time.time()
            self._loadtest_events.append(event_payload)
        except Exception:
            pass

    def get_loadtest_diagnostics(self, limit: int = 120) -> Dict[str, Any]:
        """Return summarized image load-test diagnostics for admin dashboard."""
        events = list(self._loadtest_events)[-max(1, int(limit or 120)):]
        diag_events = [item for item in events if item.get("source") == "diag"]
        attempt_events = [item for item in events if item.get("source") == "attempt"]

        def _avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 1) if values else 0.0

        def _top_stage_cpu(items: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
            stage_map: Dict[str, List[float]] = {}
            for item in items:
                stage = str(item.get("stage") or "unknown")
                value = item.get(field)
                try:
                    number = float(value)
                except Exception:
                    continue
                if number < 0:
                    continue
                stage_map.setdefault(stage, []).append(number)
            rows = []
            for stage, values in stage_map.items():
                rows.append({
                    "stage": stage,
                    "avg": _avg(values),
                    "max": round(max(values), 1) if values else 0.0,
                    "count": len(values),
                })
            rows.sort(key=lambda item: (-item.get("avg", 0), -item.get("max", 0), item.get("stage", "")))
            return rows[:6]

        process_cpu_values = [float(item.get("process_cpu_percent", -1)) for item in events if isinstance(item.get("process_cpu_percent"), (int, float)) and float(item.get("process_cpu_percent", -1)) >= 0]
        system_cpu_values = [float(item.get("system_cpu_percent", -1)) for item in events if isinstance(item.get("system_cpu_percent"), (int, float)) and float(item.get("system_cpu_percent", -1)) >= 0]
        task_values = [int(item.get("asyncio_tasks", -1)) for item in events if isinstance(item.get("asyncio_tasks"), (int, float)) and int(item.get("asyncio_tasks", -1)) >= 0]

        recaptcha_values = [float(item.get("recaptcha_ms", 0)) for item in attempt_events if isinstance(item.get("recaptcha_ms"), (int, float)) and float(item.get("recaptcha_ms", 0)) > 0]
        launch_queue_values = [float(item.get("launch_queue_ms", 0)) for item in attempt_events if isinstance(item.get("launch_queue_ms"), (int, float)) and float(item.get("launch_queue_ms", 0)) > 0]
        submit_values = [float(item.get("generate_api_ms", 0)) for item in diag_events if isinstance(item.get("generate_api_ms"), (int, float)) and float(item.get("generate_api_ms", 0)) > 0]
        upsample_values = [float(item.get("upsample_ms", 0)) for item in diag_events if isinstance(item.get("upsample_ms"), (int, float)) and float(item.get("upsample_ms", 0)) > 0]
        http_values: List[float] = []
        for item in attempt_events:
            attempts = item.get("http_attempts") or []
            if not isinstance(attempts, list):
                continue
            for http_item in attempts[:5]:
                if not isinstance(http_item, dict):
                    continue
                try:
                    duration = float(http_item.get("duration_ms", 0) or 0)
                except Exception:
                    continue
                if duration > 0:
                    http_values.append(duration)

        culprit_scores: Dict[str, float] = {}
        culprit_samples = 0
        for item in events:
            try:
                system_cpu = float(item.get("system_cpu_percent", -1) or -1)
                process_cpu = float(item.get("process_cpu_percent", -1) or -1)
            except Exception:
                continue
            if system_cpu < 70 or (process_cpu >= 0 and process_cpu >= max(25.0, system_cpu * 0.5)):
                continue
            top_processes = item.get("top_processes") or []
            if not isinstance(top_processes, list):
                continue
            culprit_samples += 1
            for proc in top_processes[:3]:
                if not isinstance(proc, dict):
                    continue
                name = str(proc.get("name") or "").strip()
                if not name:
                    continue
                try:
                    cpu_percent = float(proc.get("cpu_percent", 0) or 0)
                except Exception:
                    cpu_percent = 0.0
                culprit_scores[name] = culprit_scores.get(name, 0.0) + max(0.0, cpu_percent)

        insights: List[str] = []
        if recaptcha_values and _avg(recaptcha_values) >= max(8000.0, _avg(submit_values) * 1.2 if submit_values else 0):
            insights.append(f"reCAPTCHA 获取耗时偏高，平均 {int(_avg(recaptcha_values))} ms，优先检查打码链路/浏览器资源。")
        if submit_values and _avg(submit_values) >= max(12000.0, _avg(recaptcha_values) * 1.2 if recaptcha_values else 0):
            insights.append(f"图片提交到上游后的等待更长，平均 {int(_avg(submit_values))} ms，热点更偏上游生成或轮询等待。")
        if process_cpu_values and max(process_cpu_values) >= 70:
            insights.append(f"服务进程 CPU 峰值达到 {round(max(process_cpu_values), 1)}%，建议优先观察热点阶段排行。")
        if culprit_scores and culprit_samples > 0:
            culprit_names = [name for name, _ in sorted(culprit_scores.items(), key=lambda item: (-item[1], item[0]))[:3]]
            insights.append(
                f"整机 CPU 高而服务进程 CPU 不高的样本共 {culprit_samples} 条，最常见的高 CPU 进程为 {', '.join(culprit_names)}。"
            )
        if task_values and max(task_values) >= 200:
            insights.append(f"Asyncio 任务峰值达到 {max(task_values)}，并发调度负载较高。")
        if not insights and events:
            insights.append("当前样本中未发现单一绝对瓶颈，建议结合最近事件继续观察阶段切换。")

        recent_events = []
        for item in reversed(events[-20:]):
            recent_events.append({
                "recorded_at": item.get("recorded_at"),
                "source": item.get("source"),
                "stage": item.get("stage"),
                "request_id": item.get("request_id"),
                "token_id": item.get("token_id"),
                "process_cpu_percent": item.get("process_cpu_percent"),
                "system_cpu_percent": item.get("system_cpu_percent"),
                "asyncio_tasks": item.get("asyncio_tasks"),
                "recaptcha_ms": item.get("recaptcha_ms", 0),
                "generate_api_ms": item.get("generate_api_ms", 0),
                "upsample_ms": item.get("upsample_ms", 0),
                "error": item.get("error"),
                "top_processes": item.get("top_processes") or [],
            })

        return {
            "success": True,
            "count": len(events),
            "diag_count": len(diag_events),
            "attempt_count": len(attempt_events),
            "peaks": {
                "process_cpu_percent": round(max(process_cpu_values), 1) if process_cpu_values else 0.0,
                "system_cpu_percent": round(max(system_cpu_values), 1) if system_cpu_values else 0.0,
                "asyncio_tasks": max(task_values) if task_values else 0,
            },
            "stage_hotspots": {
                "process_cpu": _top_stage_cpu(events, "process_cpu_percent"),
                "system_cpu": _top_stage_cpu(events, "system_cpu_percent"),
            },
            "timings": {
                "recaptcha_ms_avg": _avg(recaptcha_values),
                "recaptcha_ms_max": round(max(recaptcha_values), 1) if recaptcha_values else 0.0,
                "launch_queue_ms_avg": _avg(launch_queue_values),
                "submit_api_ms_avg": _avg(submit_values),
                "submit_api_ms_max": round(max(submit_values), 1) if submit_values else 0.0,
                "http_attempt_ms_avg": _avg(http_values),
                "http_attempt_ms_max": round(max(http_values), 1) if http_values else 0.0,
                "upsample_ms_avg": _avg(upsample_values),
                "upsample_ms_max": round(max(upsample_values), 1) if upsample_values else 0.0,
            },
            "insights": insights[:6],
            "recent_events": recent_events,
        }

    @staticmethod
    def _calc_percentiles(sorted_vals: List[int]) -> Dict[str, Any]:
        """Calculate min/avg/p50/p95/p99/max from a sorted list."""
        n = len(sorted_vals)
        if n == 0:
            return {"count": 0, "min": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
        avg = round(sum(sorted_vals) / n, 1)
        return {
            "count": n,
            "min": sorted_vals[0],
            "avg": avg,
            "p50": sorted_vals[n // 2],
            "p95": sorted_vals[min(int(n * 0.95), n - 1)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
            "max": sorted_vals[-1],
        }

    @staticmethod
    def _build_timeline(
        records: List[RequestRecord],
        now: float,
        bucket_seconds: int = 60,
        buckets: int = 5,
    ) -> List[Dict[str, Any]]:
        """Build per-minute throughput timeline."""
        result = []
        for i in range(buckets - 1, -1, -1):
            bucket_end = now - i * bucket_seconds
            bucket_start = bucket_end - bucket_seconds
            bucket_records = [
                r for r in records
                if r.finished_at and bucket_start <= r.finished_at < bucket_end
            ]
            success = sum(1 for r in bucket_records if r.status == "success")
            failed = sum(1 for r in bucket_records if r.status == "failed")
            durations = [r.duration_ms for r in bucket_records if r.duration_ms is not None]
            avg_dur = round(sum(durations) / len(durations), 0) if durations else 0
            minute_label = time.strftime("%H:%M", time.localtime(bucket_end))
            result.append({
                "label": minute_label,
                "total": len(bucket_records),
                "success": success,
                "failed": failed,
                "avg_duration_ms": avg_dur,
            })
        return result


# Lightweight system resource helper — kept inline to avoid external deps
class _ResourceInfo:
    _fallback_last_cpu_time: Optional[float] = None
    _fallback_last_wall_time: Optional[float] = None
    _system_cpu_primed: bool = False
    _system_last_idle_100ns: Optional[int] = None
    _system_last_total_100ns: Optional[int] = None
    _top_process_cache: List[Dict[str, Any]] = []
    _top_process_cache_at: float = 0.0
    _top_process_last_sample_at: Optional[float] = None
    _top_process_cpu_times: Dict[int, float] = {}

    @staticmethod
    def _logical_cpu_count() -> int:
        return max(1, os.cpu_count() or 1)

    @classmethod
    def prime_system_cpu_counter(cls) -> None:
        if cls._system_cpu_primed:
            return
        try:
            psutil = importlib.import_module("psutil")
            psutil.cpu_percent(interval=None)
            cls._system_cpu_primed = True
        except Exception:
            cls._system_cpu_primed = cls._prime_native_system_cpu_counter()

    @staticmethod
    def _filetime_to_int(filetime: Any) -> int:
        return (int(getattr(filetime, "dwHighDateTime", 0)) << 32) | int(getattr(filetime, "dwLowDateTime", 0))

    @classmethod
    def _prime_native_system_cpu_counter(cls) -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes
            from ctypes import wintypes

            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetSystemTimes.argtypes = [
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
            ]
            kernel32.GetSystemTimes.restype = wintypes.BOOL

            idle = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            ok = kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            if not ok:
                return False
            idle_val = cls._filetime_to_int(idle)
            total_val = cls._filetime_to_int(kernel) + cls._filetime_to_int(user)
            cls._system_last_idle_100ns = idle_val
            cls._system_last_total_100ns = total_val
            return True
        except Exception:
            return False

    @classmethod
    def _normalize_process_cpu_percent(cls, value: float) -> float:
        try:
            cpu = float(value)
        except Exception:
            return -1.0
        if cpu < 0:
            return -1.0
        return round(max(0.0, cpu) / cls._logical_cpu_count(), 1)

    @staticmethod
    def _safe_asyncio_task_count() -> int:
        try:
            return len(asyncio.all_tasks())
        except RuntimeError:
            return 0
        except Exception:
            return -1

    @classmethod
    def get_top_processes(cls, limit: int = 5, cache_ttl_seconds: float = 1.5) -> List[Dict[str, Any]]:
        """Return top system processes by CPU usage with a short cache."""
        normalized_limit = max(1, int(limit or 5))
        now = time.monotonic()
        if cls._top_process_cache and (now - cls._top_process_cache_at) < max(0.2, float(cache_ttl_seconds or 1.5)):
            return [dict(item) for item in cls._top_process_cache[:normalized_limit]]

        try:
            psutil = importlib.import_module("psutil")
        except Exception:
            return []

        current_sample_at = time.perf_counter()
        elapsed = None
        if cls._top_process_last_sample_at is not None:
            elapsed = max(current_sample_at - cls._top_process_last_sample_at, 1e-6)
        seen_cpu_times: Dict[int, float] = {}
        rows: List[Dict[str, Any]] = []

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                pid = int(getattr(proc, "pid", 0) or 0)
                if pid <= 0:
                    continue
                cpu_times = proc.cpu_times()
                total_cpu_time = float(getattr(cpu_times, "user", 0.0) or 0.0) + float(getattr(cpu_times, "system", 0.0) or 0.0)
                seen_cpu_times[pid] = total_cpu_time

                cpu_percent = 0.0
                previous_cpu_time = cls._top_process_cpu_times.get(pid)
                if previous_cpu_time is not None and elapsed is not None:
                    delta_cpu = max(0.0, total_cpu_time - previous_cpu_time)
                    cpu_percent = min(
                        100.0,
                        max(0.0, (delta_cpu / elapsed) * 100.0 / cls._logical_cpu_count()),
                    )

                rss_mb = -1.0
                try:
                    rss_mb = round(float(proc.memory_info().rss or 0) / 1024 / 1024, 1)
                except Exception:
                    pass

                name = str(getattr(proc, "info", {}).get("name") or "") if hasattr(proc, "info") else ""
                if not name:
                    try:
                        name = str(proc.name() or "")
                    except Exception:
                        name = ""
                rows.append({
                    "pid": pid,
                    "name": name or f"pid-{pid}",
                    "cpu_percent": round(cpu_percent, 1),
                    "memory_rss_mb": rss_mb,
                    "is_current_process": pid == os.getpid(),
                })
            except Exception:
                continue

        cls._top_process_last_sample_at = current_sample_at
        cls._top_process_cpu_times = seen_cpu_times
        rows.sort(
            key=lambda item: (
                -float(item.get("cpu_percent", 0.0) or 0.0),
                -float(item.get("memory_rss_mb", -1.0) or -1.0),
                str(item.get("name") or ""),
            )
        )
        cls._top_process_cache = rows[: max(normalized_limit, 8)]
        cls._top_process_cache_at = now
        return [dict(item) for item in cls._top_process_cache[:normalized_limit]]

    @classmethod
    def _get_native_process_info(cls) -> Dict[str, Any]:
        cpu_percent = -1.0
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        if cls._fallback_last_cpu_time is not None and cls._fallback_last_wall_time is not None:
            elapsed_wall = max(now_wall - cls._fallback_last_wall_time, 1e-6)
            elapsed_cpu = max(now_cpu - cls._fallback_last_cpu_time, 0.0)
            cpu_percent = min(100.0, max(0.0, (elapsed_cpu / elapsed_wall) * 100.0 / cls._logical_cpu_count()))
        else:
            cpu_percent = 0.0
        cls._fallback_last_cpu_time = now_cpu
        cls._fallback_last_wall_time = now_wall

        memory_rss_mb = -1.0
        memory_vms_mb = -1.0
        try:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                kernel32.GetCurrentProcess.restype = wintypes.HANDLE
                psapi.GetProcessMemoryInfo.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                    wintypes.DWORD,
                ]
                psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                handle = kernel32.GetCurrentProcess()
                ok = psapi.GetProcessMemoryInfo(
                    handle,
                    ctypes.byref(counters),
                    counters.cb,
                )
                if ok:
                    memory_rss_mb = round(counters.WorkingSetSize / 1024 / 1024, 1)
                    memory_vms_mb = round(counters.PagefileUsage / 1024 / 1024, 1)
            else:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                rss = float(getattr(usage, "ru_maxrss", 0) or 0)
                # macOS ru_maxrss is bytes; Linux is KB.
                if sys.platform == "darwin":
                    memory_rss_mb = round(rss / 1024 / 1024, 1)
                else:
                    memory_rss_mb = round(rss / 1024, 1)
        except Exception:
            pass

        return {
            "pid": os.getpid(),
            "memory_rss_mb": memory_rss_mb,
            "memory_vms_mb": memory_vms_mb,
            "cpu_percent": round(cpu_percent, 1) if cpu_percent >= 0 else -1,
            "cpu_percent_raw": -1,
            "threads": threading.active_count(),
            "open_files": -1,
            "source": "native",
            "detail": "native runtime fallback",
            "cpu_semantics": "task_manager",
        }

    @classmethod
    def get_system_snapshot(cls) -> Dict[str, Any]:
        sampled_at = time.time()
        try:
            psutil = importlib.import_module("psutil")
            if not cls._system_cpu_primed:
                psutil.cpu_percent(interval=None)
                cls._system_cpu_primed = True
            cpu_percent = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            return {
                "success": True,
                "sampled_at": sampled_at,
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(float(getattr(vm, "percent", -1) or -1), 1),
                "memory_used_mb": round(float(getattr(vm, "used", 0) or 0) / 1024 / 1024, 1),
                "memory_total_mb": round(float(getattr(vm, "total", 0) or 0) / 1024 / 1024, 1),
                "logical_cpu_count": int(psutil.cpu_count(logical=True) or cls._logical_cpu_count()),
                "physical_cpu_count": int(psutil.cpu_count(logical=False) or 0),
                "source": "system",
                "cpu_semantics": "system_total",
            }
        except Exception as exc:
            native_snapshot = cls._get_native_system_snapshot()
            if native_snapshot.get("success"):
                native_snapshot["detail"] = f"psutil unavailable, using native system snapshot: {str(exc) or exc.__class__.__name__}"
                return native_snapshot
            return {
                "success": False,
                "sampled_at": sampled_at,
                "cpu_percent": -1,
                "memory_percent": -1,
                "memory_used_mb": -1,
                "memory_total_mb": -1,
                "logical_cpu_count": cls._logical_cpu_count(),
                "physical_cpu_count": -1,
                "source": "system",
                "detail": str(exc) or exc.__class__.__name__,
                "cpu_semantics": "system_total",
            }

    @classmethod
    def _get_native_system_snapshot(cls) -> Dict[str, Any]:
        sampled_at = time.time()
        if os.name != "nt":
            return {
                "success": False,
                "sampled_at": sampled_at,
                "cpu_percent": -1,
                "memory_percent": -1,
                "memory_used_mb": -1,
                "memory_total_mb": -1,
                "logical_cpu_count": cls._logical_cpu_count(),
                "physical_cpu_count": -1,
                "source": "native-system",
                "detail": "native system snapshot unsupported on this platform",
                "cpu_semantics": "system_total",
            }
        try:
            import ctypes
            from ctypes import wintypes

            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD),
                ]

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetSystemTimes.argtypes = [
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
            ]
            kernel32.GetSystemTimes.restype = wintypes.BOOL
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
            kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL

            idle = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                raise OSError(f"GetSystemTimes failed: {ctypes.get_last_error()}")

            idle_val = cls._filetime_to_int(idle)
            total_val = cls._filetime_to_int(kernel) + cls._filetime_to_int(user)
            cpu_percent = 0.0
            if cls._system_last_idle_100ns is not None and cls._system_last_total_100ns is not None:
                idle_delta = max(0, idle_val - cls._system_last_idle_100ns)
                total_delta = max(1, total_val - cls._system_last_total_100ns)
                cpu_percent = max(0.0, min(100.0, (1.0 - (idle_delta / total_delta)) * 100.0))
            cls._system_last_idle_100ns = idle_val
            cls._system_last_total_100ns = total_val

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                raise OSError(f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}")

            total_phys = float(mem.ullTotalPhys or 0)
            avail_phys = float(mem.ullAvailPhys or 0)
            used_phys = max(0.0, total_phys - avail_phys)
            return {
                "success": True,
                "sampled_at": sampled_at,
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(float(mem.dwMemoryLoad or 0), 1),
                "memory_used_mb": round(used_phys / 1024 / 1024, 1),
                "memory_total_mb": round(total_phys / 1024 / 1024, 1),
                "logical_cpu_count": cls._logical_cpu_count(),
                "physical_cpu_count": -1,
                "source": "native-system",
                "cpu_semantics": "system_total",
            }
        except Exception as exc:
            return {
                "success": False,
                "sampled_at": sampled_at,
                "cpu_percent": -1,
                "memory_percent": -1,
                "memory_used_mb": -1,
                "memory_total_mb": -1,
                "logical_cpu_count": cls._logical_cpu_count(),
                "physical_cpu_count": -1,
                "source": "native-system",
                "detail": str(exc) or exc.__class__.__name__,
                "cpu_semantics": "system_total",
            }

    @staticmethod
    def get_info() -> Dict[str, Any]:
        try:
            psutil = importlib.import_module("psutil")
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            cpu_pct_raw = float(proc.cpu_percent(interval=None))
            cpu_pct = _ResourceInfo._normalize_process_cpu_percent(cpu_pct_raw)
            return {
                "pid": os.getpid(),
                "memory_rss_mb": round(mem.rss / 1024 / 1024, 1),
                "memory_vms_mb": round(mem.vms / 1024 / 1024, 1),
                "cpu_percent": cpu_pct,
                "cpu_percent_raw": round(cpu_pct_raw, 1),
                "threads": proc.num_threads(),
                "open_files": len(proc.open_files()) if hasattr(proc, "open_files") else -1,
                "asyncio_tasks": _ResourceInfo._safe_asyncio_task_count(),
                "source": "psutil",
                "cpu_semantics": "task_manager",
            }
        except ImportError:
            result = _ResourceInfo._get_native_process_info()
            result["asyncio_tasks"] = _ResourceInfo._safe_asyncio_task_count()
            return result
        except Exception:
            result = _ResourceInfo._get_native_process_info()
            result["asyncio_tasks"] = _ResourceInfo._safe_asyncio_task_count() if asyncio else -1
            return result

    @staticmethod
    def get_network_counters() -> Optional[Dict[str, int]]:
        try:
            import importlib
            psutil = importlib.import_module("psutil")
            counters = psutil.net_io_counters()
            return {
                "bytes_sent": int(getattr(counters, "bytes_sent", 0) or 0),
                "bytes_recv": int(getattr(counters, "bytes_recv", 0) or 0),
            }
        except Exception:
            return None


# Global singleton
perf_monitor = PerfMonitor()
