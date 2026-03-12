"""Lightweight in-memory performance monitor for high-concurrency diagnostics.

Collects rolling metrics with minimal overhead. All data is ephemeral
(lost on restart) — this is intentional to avoid adding DB I/O pressure.
"""

import asyncio
import os
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
                })
            inflight_list.sort(key=lambda x: -x["elapsed_ms"])

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
    @staticmethod
    def get_info() -> Dict[str, Any]:
        try:
            import importlib
            psutil = importlib.import_module("psutil")
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            cpu_pct = proc.cpu_percent(interval=0)
            return {
                "pid": os.getpid(),
                "memory_rss_mb": round(mem.rss / 1024 / 1024, 1),
                "memory_vms_mb": round(mem.vms / 1024 / 1024, 1),
                "cpu_percent": cpu_pct,
                "threads": proc.num_threads(),
                "open_files": len(proc.open_files()) if hasattr(proc, "open_files") else -1,
                "asyncio_tasks": len(asyncio.all_tasks()),
            }
        except ImportError:
            # psutil not installed — fallback
            return {
                "pid": os.getpid(),
                "memory_rss_mb": -1,
                "memory_vms_mb": -1,
                "cpu_percent": -1,
                "threads": -1,
                "open_files": -1,
                "asyncio_tasks": len(asyncio.all_tasks()),
            }
        except Exception:
            return {
                "pid": os.getpid(),
                "asyncio_tasks": len(asyncio.all_tasks()) if asyncio else -1,
            }

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
