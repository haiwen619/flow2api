"""Cluster coordination for Flow2API master/worker deployments."""
import asyncio
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from ..core.config import config
from ..core.logger import debug_logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_unreachable_cluster_base_url(base_url: str) -> bool:
    try:
        parsed = urlparse(str(base_url or "").strip())
    except Exception:
        return False
    hostname = str(parsed.hostname or "").strip().lower()
    return hostname in {"0.0.0.0", "::", "[::]"}


def _normalize_worker_token_usage_rows(rows: Any) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        email = str(row.get("email") or "").strip().lower()
        if not email:
            continue
        token_id_raw = row.get("token_id")
        token_id: Optional[int] = None
        if token_id_raw is not None:
            try:
                token_id = int(token_id_raw)
            except Exception:
                token_id = None
        normalized_rows.append(
            {
                "token_id": token_id,
                "email": email,
                "current_project_id": str(row.get("current_project_id") or "").strip() or None,
                "current_project_name": str(row.get("current_project_name") or "").strip() or None,
                "remark": str(row.get("remark") or "").strip() or None,
            }
        )
    return normalized_rows


class ClusterManager:
    """Coordinate worker heartbeats and request dispatch decisions."""

    def __init__(self, load_balancer=None):
        self.load_balancer = load_balancer
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._nodes_lock = asyncio.Lock()
        self._dispatch_stats: Dict[str, Dict[str, Any]] = {}
        self._dispatch_stats_lock = asyncio.Lock()
        self._delegated_auto_login_jobs: Dict[str, Dict[str, Any]] = {}
        self._delegated_auto_login_lock = asyncio.Lock()
        self._local_active_requests = 0
        self._local_lock = asyncio.Lock()
        self._inflight_dispatches: Dict[str, int] = {}  # node_id → 主节点已发出但未完成的请求数
        self._worker_heartbeat_task: Optional[asyncio.Task] = None
        self._master_cleanup_task: Optional[asyncio.Task] = None
        self._last_master_sync: Dict[str, Any] = {
            "success": False,
            "last_attempt_at": None,
            "last_success_at": None,
            "last_error": None,
            "roundtrip_ms": None,
        }

    async def record_dispatch(
        self,
        *,
        target_node_id: str,
        target_node_name: Optional[str] = None,
        generation_type: Optional[str] = None,
        resolved_model: Optional[str] = None,
        status_code: Optional[int] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        node_id = str(target_node_id or "").strip()
        if not node_id:
            return

        # 释放飞行中计数（在主节点收到响应时调用）
        inflight = self._inflight_dispatches.get(node_id, 0)
        if inflight > 1:
            self._inflight_dispatches[node_id] = inflight - 1
        elif inflight == 1:
            self._inflight_dispatches.pop(node_id, None)

        is_success = error is None and (status_code is None or int(status_code) < 400)
        now_iso = _utc_now_iso()
        async with self._dispatch_stats_lock:
            current = dict(self._dispatch_stats.get(node_id) or {})
            current["node_id"] = node_id
            current["node_name"] = str(target_node_name or current.get("node_name") or node_id).strip()
            current["dispatch_count"] = int(current.get("dispatch_count") or 0) + 1
            current["success_count"] = int(current.get("success_count") or 0) + (1 if is_success else 0)
            current["failure_count"] = int(current.get("failure_count") or 0) + (0 if is_success else 1)
            current["image_count"] = int(current.get("image_count") or 0) + (1 if generation_type == "image" else 0)
            current["video_count"] = int(current.get("video_count") or 0) + (1 if generation_type == "video" else 0)
            current["total_duration_ms"] = int(current.get("total_duration_ms") or 0) + max(0, int(duration_ms or 0))
            current["last_status_code"] = status_code
            current["last_error"] = str(error or "").strip() or None
            current["last_model"] = str(resolved_model or "").strip() or None
            current["last_generation_type"] = str(generation_type or "").strip() or None
            current["last_dispatch_at"] = now_iso
            self._dispatch_stats[node_id] = current

    async def _build_dispatch_stats_snapshot(self) -> Dict[str, Dict[str, Any]]:
        async with self._dispatch_stats_lock:
            return {str(node_id): dict(item) for node_id, item in self._dispatch_stats.items()}

    async def record_delegated_auto_login_job(
        self,
        *,
        delegation_id: str,
        status: str,
        detail: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        key = str(delegation_id or "").strip()
        if not key:
            raise ValueError("delegation_id 不能为空")

        now_iso = _utc_now_iso()
        async with self._delegated_auto_login_lock:
            current = dict(self._delegated_auto_login_jobs.get(key) or {})
            if not current:
                current["delegation_id"] = key
                current["created_at"] = now_iso

            for field_name, field_value in fields.items():
                if field_value is None:
                    continue
                current[field_name] = field_value

            current["status"] = str(status or current.get("status") or "queued").strip().lower()
            current["detail"] = str(detail or current.get("detail") or "").strip() or None
            current["updated_at"] = now_iso
            self._delegated_auto_login_jobs[key] = current

            if len(self._delegated_auto_login_jobs) > 200:
                ordered = sorted(
                    self._delegated_auto_login_jobs.items(),
                    key=lambda item: str(item[1].get("created_at") or ""),
                    reverse=True,
                )
                self._delegated_auto_login_jobs = dict(ordered[:200])

            return dict(self._delegated_auto_login_jobs[key])

    async def list_delegated_auto_login_jobs(self, limit: int = 30) -> List[Dict[str, Any]]:
        try:
            max_items = max(1, min(int(limit or 30), 100))
        except Exception:
            max_items = 30

        async with self._delegated_auto_login_lock:
            rows = [dict(item) for item in self._delegated_auto_login_jobs.values()]

        rows.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return rows[:max_items]

    @property
    def role(self) -> str:
        return config.cluster_role

    def is_enabled(self) -> bool:
        return bool(config.cluster_enabled)

    def is_master(self) -> bool:
        return self.role == "master"

    def is_worker(self) -> bool:
        return self.role == "worker"

    def should_dispatch_externally(self) -> bool:
        return self.is_master() and self.is_enabled()

    def verify_cluster_key(self, provided_key: Optional[str]) -> bool:
        expected = str(config.cluster_key or "").strip()
        actual = str(provided_key or "").strip()
        if not expected or not actual:
            return False
        return secrets.compare_digest(expected, actual)

    async def start(self):
        await self.stop()
        if not self.is_enabled():
            return

        if self.is_master():
            self._master_cleanup_task = asyncio.create_task(self._master_cleanup_loop())
            debug_logger.log_info("[CLUSTER] 主节点清理任务已启动")

        if self.is_worker():
            self._worker_heartbeat_task = asyncio.create_task(self._worker_heartbeat_loop())
            debug_logger.log_info("[CLUSTER] 子节点心跳任务已启动")

    async def stop(self):
        for task in (self._worker_heartbeat_task, self._master_cleanup_task):
            if task:
                task.cancel()
        for task in (self._worker_heartbeat_task, self._master_cleanup_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._worker_heartbeat_task = None
        self._master_cleanup_task = None

    async def reload_runtime(self):
        await self.start()

    async def acquire_local_slot(self) -> bool:
        async with self._local_lock:
            if not self.is_enabled():
                self._local_active_requests += 1
                return True
            limit = max(0, int(config.cluster_node_max_concurrency))
            if limit > 0 and self._local_active_requests >= limit:
                return False
            self._local_active_requests += 1
            return True

    async def release_local_slot(self):
        async with self._local_lock:
            self._local_active_requests = max(0, self._local_active_requests - 1)

    async def get_local_active_requests(self) -> int:
        async with self._local_lock:
            return int(self._local_active_requests)

    async def _build_load_summary(self) -> Dict[str, Any]:
        if not self.load_balancer:
            return {
                "token_count": 0,
                "total_image_inflight": 0,
                "total_video_inflight": 0,
                "total_image_capacity": None,
                "total_video_capacity": None,
            }

        try:
            status = await self.load_balancer.get_all_load_status()
        except Exception as exc:
            debug_logger.log_warning(f"[CLUSTER] 获取本地负载快照失败: {exc}")
            return {
                "token_count": 0,
                "total_image_inflight": 0,
                "total_video_inflight": 0,
                "total_image_capacity": None,
                "total_video_capacity": None,
            }

        return dict(status.get("summary") or {})

    async def _build_local_worker_token_usage(self) -> List[Dict[str, Any]]:
        if not self.is_worker() or not self.load_balancer or not getattr(self.load_balancer, "token_manager", None):
            return []
        try:
            active_tokens = await self.load_balancer.token_manager.get_active_tokens()
        except Exception as exc:
            debug_logger.log_warning(f"[CLUSTER] 获取子节点活跃 Token 失败: {exc}")
            return []

        rows: List[Dict[str, Any]] = []
        for token in active_tokens:
            email = str(getattr(token, "email", "") or "").strip().lower()
            if not email:
                continue
            rows.append(
                {
                    "token_id": getattr(token, "id", None),
                    "email": email,
                    "current_project_id": getattr(token, "current_project_id", None),
                    "current_project_name": getattr(token, "current_project_name", None),
                    "remark": getattr(token, "remark", None),
                }
            )
        return rows

    async def build_local_node_snapshot(self) -> Dict[str, Any]:
        active_requests = await self.get_local_active_requests()
        node_max = max(0, int(config.cluster_node_max_concurrency))
        load_summary = await self._build_load_summary()
        available_slots = max(0, node_max - active_requests) if node_max > 0 else None
        return {
            "node_id": config.cluster_node_id,
            "node_name": config.cluster_node_name,
            "base_url": config.cluster_effective_node_public_base_url,
            "server_port": int(config.server_port),
            "worker_tokens": await self._build_local_worker_token_usage(),
            "role": self.role,
            "enabled": True,
            "healthy": True,
            "active_requests": active_requests,
            "node_max_concurrency": node_max,
            "available_slots": available_slots,
            "weight": int(config.cluster_node_weight),
            "load_summary": load_summary,
            "status_reason": "local",
            "last_seen_at": _utc_now_iso(),
            "heartbeat_age_seconds": 0,
            "reported_roundtrip_ms": (
                self._last_master_sync.get("roundtrip_ms") if self.is_worker() else 0
            ),
        }

    async def register_heartbeat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = str(payload.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("node_id 不能为空")

        now = time.time()
        sanitized = {
            "node_id": node_id,
            "node_name": str(payload.get("node_name") or node_id).strip(),
            "base_url": str(payload.get("base_url") or "").strip().rstrip("/"),
            "server_port": max(1, int(payload.get("server_port") or 0)) if payload.get("server_port") is not None else None,
            "worker_tokens": _normalize_worker_token_usage_rows(payload.get("worker_tokens") or []),
            "role": str(payload.get("role") or "worker").strip().lower(),
            "reported_enabled": bool(payload.get("enabled", True)),
            "enabled": bool(payload.get("enabled", True)),
            "active_requests": max(0, int(payload.get("active_requests") or 0)),
            "node_max_concurrency": max(0, int(payload.get("node_max_concurrency") or 0)),
            "weight": max(1, int(payload.get("weight") or 100)),
            "available_slots": payload.get("available_slots"),
            "load_summary": dict(payload.get("load_summary") or {}),
            "reported_roundtrip_ms": payload.get("reported_roundtrip_ms"),
            "reported_at": str(payload.get("reported_at") or _utc_now_iso()),
            "last_seen_ts": now,
            "last_seen_at": _utc_now_iso(),
        }
        if sanitized["available_slots"] is not None:
            try:
                sanitized["available_slots"] = max(0, int(sanitized["available_slots"]))
            except Exception:
                sanitized["available_slots"] = None

        async with self._nodes_lock:
            existing = dict(self._nodes.get(node_id) or {})
            manual_override = existing.get("manual_enabled_override")
            existing.update(sanitized)
            if manual_override is not None:
                existing["manual_enabled_override"] = bool(manual_override)
                existing["enabled"] = bool(manual_override)
            self._nodes[node_id] = existing

        return {"success": True, "received_at": _utc_now_iso(), "node_id": node_id}

    async def get_worker_token_usage_map(self) -> Dict[str, List[Dict[str, Any]]]:
        now_ts = time.time()
        async with self._nodes_lock:
            remote_nodes = [dict(item) for item in self._nodes.values()]

        usage_map: Dict[str, List[Dict[str, Any]]] = {}
        for raw_node in remote_nodes:
            node = self._decorate_node(raw_node, now_ts)
            if str(node.get("role") or "").strip().lower() != "worker":
                continue
            if not bool(node.get("healthy")) or not bool(node.get("enabled", True)):
                continue

            for usage in _normalize_worker_token_usage_rows(node.get("worker_tokens") or []):
                email = str(usage.get("email") or "").strip().lower()
                if not email:
                    continue
                usage_item = dict(usage)
                usage_item["node_id"] = str(node.get("node_id") or "").strip()
                usage_item["node_name"] = str(node.get("node_name") or usage_item["node_id"]).strip()
                usage_item["reported_at"] = node.get("reported_at") or node.get("last_seen_at")
                usage_map.setdefault(email, []).append(usage_item)

        for email, rows in usage_map.items():
            rows.sort(key=lambda item: str(item.get("reported_at") or ""), reverse=True)
        return usage_map

    async def get_worker_used_token_emails(self) -> set[str]:
        usage_map = await self.get_worker_token_usage_map()
        return {email for email, rows in usage_map.items() if rows}

    async def is_token_email_used_by_worker(self, email: Optional[str]) -> bool:
        normalized = str(email or "").strip().lower()
        if not normalized:
            return False
        usage_map = await self.get_worker_token_usage_map()
        return bool(usage_map.get(normalized))

    async def remove_node(self, node_id: str) -> bool:
        """Remove a remote node record from the master's in-memory registry."""
        target = str(node_id or "").strip()
        if not target:
            raise ValueError("node_id 不能为空")
        if target == config.cluster_node_id:
            raise ValueError("不能删除当前节点")

        async with self._nodes_lock:
            existed = target in self._nodes
            if existed:
                self._nodes.pop(target, None)
        return existed

    async def set_node_enabled(self, node_id: str, enabled: bool) -> bool:
        """Manually enable or disable a remote node."""
        target = str(node_id or "").strip()
        if not target:
            raise ValueError("node_id 不能为空")
        if target == config.cluster_node_id:
            raise ValueError("不能禁用当前节点")

        async with self._nodes_lock:
            current = dict(self._nodes.get(target) or {})
            if not current:
                return False
            current["manual_enabled_override"] = bool(enabled)
            current["enabled"] = bool(enabled)
            current["last_manual_override_at"] = _utc_now_iso()
            self._nodes[target] = current
        return True

    async def _master_cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(max(5, int(config.cluster_heartbeat_interval_seconds)))
                await self.prune_stale_nodes()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                debug_logger.log_warning(f"[CLUSTER] 主节点清理异常: {exc}")

    async def prune_stale_nodes(self):
        timeout_seconds = max(5, int(config.cluster_heartbeat_timeout_seconds))
        now = time.time()
        async with self._nodes_lock:
            stale_node_ids = [
                node_id
                for node_id, node in self._nodes.items()
                if (now - float(node.get("last_seen_ts") or 0)) > (timeout_seconds * 3)
            ]
            for node_id in stale_node_ids:
                self._nodes.pop(node_id, None)

    async def _worker_heartbeat_loop(self):
        while True:
            try:
                await self._send_heartbeat_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_master_sync.update(
                    {
                        "success": False,
                        "last_attempt_at": _utc_now_iso(),
                        "last_error": str(exc),
                    }
                )
                debug_logger.log_warning(f"[CLUSTER] 子节点心跳失败: {exc}")

            await asyncio.sleep(max(3, int(config.cluster_heartbeat_interval_seconds)))

    async def _send_heartbeat_once(self):
        master_base_url = str(config.cluster_master_base_url or "").strip().rstrip("/")
        cluster_key = str(config.cluster_key or "").strip()
        if not master_base_url or not cluster_key:
            raise RuntimeError("worker 模式下必须配置 cluster.master_base_url 与 cluster.cluster_key")

        payload = await self.build_local_node_snapshot()
        payload["reported_at"] = _utc_now_iso()
        payload["reported_roundtrip_ms"] = self._last_master_sync.get("roundtrip_ms")
        endpoint = f"{master_base_url}/api/internal/cluster/heartbeat"
        started_at = time.perf_counter()

        async with httpx.AsyncClient(timeout=max(5, int(config.cluster_heartbeat_timeout_seconds))) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers={
                    "X-Cluster-Key": cluster_key,
                    "X-Cluster-Node-Id": str(payload.get("node_id") or ""),
                },
            )

        roundtrip_ms = int((time.perf_counter() - started_at) * 1000)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        self._last_master_sync.update(
            {
                "success": True,
                "last_attempt_at": _utc_now_iso(),
                "last_success_at": _utc_now_iso(),
                "last_error": None,
                "roundtrip_ms": roundtrip_ms,
            }
        )

    def _decorate_node(self, node: Dict[str, Any], now_ts: float) -> Dict[str, Any]:
        item = dict(node or {})
        last_seen_ts = float(item.get("last_seen_ts") or now_ts)
        heartbeat_age = max(0, int(now_ts - last_seen_ts))
        timeout_seconds = max(5, int(config.cluster_heartbeat_timeout_seconds))
        effective_enabled = bool(item.get("enabled", True))
        healthy = effective_enabled and heartbeat_age <= timeout_seconds

        item["enabled"] = effective_enabled
        item["reported_enabled"] = bool(item.get("reported_enabled", effective_enabled))
        item["healthy"] = healthy
        item["heartbeat_age_seconds"] = heartbeat_age
        item["available_slots"] = (
            max(0, int(item.get("node_max_concurrency") or 0) - int(item.get("active_requests") or 0))
            if item.get("node_max_concurrency") is not None and int(item.get("node_max_concurrency") or 0) > 0
            else item.get("available_slots")
        )
        if not item.get("status_reason"):
            if not effective_enabled:
                item["status_reason"] = "manual_disabled"
            elif healthy:
                item["status_reason"] = "healthy"
            else:
                item["status_reason"] = "heartbeat_timeout"
        return item

    async def list_nodes(self) -> List[Dict[str, Any]]:
        now_ts = time.time()
        nodes: List[Dict[str, Any]] = []

        local = await self.build_local_node_snapshot()
        local["healthy"] = True
        nodes.append(local)

        async with self._nodes_lock:
            remote_nodes = [dict(item) for item in self._nodes.values()]

        for item in remote_nodes:
            nodes.append(self._decorate_node(item, now_ts))

        nodes.sort(key=lambda item: (0 if item.get("role") == "master" else 1, str(item.get("node_name") or "")))
        return nodes

    async def choose_dispatch_target(
        self,
        *,
        resolved_model: Optional[str],
        generation_type: Optional[str],
    ) -> Dict[str, Any]:
        if not self.should_dispatch_externally():
            return {"dispatch_to": "local", "reason": "not_master"}

        nodes = await self.list_nodes()
        healthy_nodes = [
            node for node in nodes
            if bool(node.get("healthy")) and bool(node.get("enabled", True))
        ]
        if not healthy_nodes:
            return {"dispatch_to": "local", "reason": "no_healthy_nodes"}

        # 快照当前飞行中的请求数（asyncio 单线程，无 await 的赋值操作是原子的）
        inflight_snapshot = dict(self._inflight_dispatches)

        candidates: List[Dict[str, Any]] = []
        local_candidate: Optional[Dict[str, Any]] = None
        for node in healthy_nodes:
            node_id_key = str(node.get("node_id") or "")
            node_max = max(0, int(node.get("node_max_concurrency") or 0))
            active = max(0, int(node.get("active_requests") or 0))
            if node_max <= 0:
                available = 0
            else:
                # 减去已发出但尚未完成的飞行请求，防止心跳过期期间重复超发
                inflight = inflight_snapshot.get(node_id_key, 0)
                available = max(0, node_max - active - inflight)
            if available <= 0:
                continue

            weight = max(1, int(node.get("weight") or 100))
            weighted_available = available * weight
            if str(node.get("node_id") or "") == config.cluster_node_id and bool(config.cluster_prefer_local):
                weighted_available += max(1, weight // 10)

            candidate = dict(node)
            candidate["score"] = weighted_available
            candidate["dispatch_generation_type"] = generation_type
            candidate["dispatch_model"] = resolved_model
            candidates.append(candidate)
            if str(candidate.get("node_id") or "") == config.cluster_node_id:
                local_candidate = candidate

        if not candidates:
            return {"dispatch_to": "local", "reason": "all_nodes_full"}

        if bool(config.cluster_prefer_local) and local_candidate is not None:
            return {"dispatch_to": "local", "reason": "prefer_local_available", "node": local_candidate}

        selected = max(
            candidates,
            key=lambda item: (
                int(item.get("score") or 0),
                int(item.get("available_slots") or 0),
                -int(item.get("active_requests") or 0),
                -int(item.get("reported_roundtrip_ms") or 0),
            ),
        )

        if str(selected.get("node_id") or "") == config.cluster_node_id:
            return {"dispatch_to": "local", "reason": "selected_local", "node": selected}

        selected_base_url = str(selected.get("base_url") or "").strip()
        if not selected_base_url:
            return {"dispatch_to": "local", "reason": "selected_remote_missing_base_url"}
        if _is_unreachable_cluster_base_url(selected_base_url):
            return {"dispatch_to": "local", "reason": "selected_remote_unreachable_base_url", "node": selected}

        # 记录飞行中请求，防止下次 choose_dispatch_target 再次超发到同一节点
        selected_node_id = str(selected.get("node_id") or "")
        if selected_node_id:
            self._inflight_dispatches[selected_node_id] = (
                self._inflight_dispatches.get(selected_node_id, 0) + 1
            )

        return {"dispatch_to": "remote", "reason": "selected_remote", "node": selected}

    async def get_cluster_snapshot(self) -> Dict[str, Any]:
        nodes = await self.list_nodes()
        dispatch_stats_map = await self._build_dispatch_stats_snapshot()
        remote_dispatch_total = 0
        remote_dispatch_success = 0
        remote_dispatch_failure = 0

        for node in nodes:
            node_id = str(node.get("node_id") or "").strip()
            stats = dict(dispatch_stats_map.get(node_id) or {})
            dispatch_count = int(stats.get("dispatch_count") or 0)
            success_count = int(stats.get("success_count") or 0)
            failure_count = int(stats.get("failure_count") or 0)
            total_duration_ms = int(stats.get("total_duration_ms") or 0)
            avg_duration_ms = int(total_duration_ms / dispatch_count) if dispatch_count > 0 else None
            node["dispatch_stats"] = {
                "dispatch_count": dispatch_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "image_count": int(stats.get("image_count") or 0),
                "video_count": int(stats.get("video_count") or 0),
                "avg_duration_ms": avg_duration_ms,
                "last_dispatch_at": stats.get("last_dispatch_at"),
                "last_status_code": stats.get("last_status_code"),
                "last_error": stats.get("last_error"),
                "last_model": stats.get("last_model"),
                "last_generation_type": stats.get("last_generation_type"),
            }
            if node_id != str(config.cluster_node_id or "").strip() and str(node.get("role") or "") == "worker":
                remote_dispatch_total += dispatch_count
                remote_dispatch_success += success_count
                remote_dispatch_failure += failure_count

        for node in nodes:
            stats = dict(node.get("dispatch_stats") or {})
            dispatch_count = int(stats.get("dispatch_count") or 0)
            stats["traffic_share"] = (
                round((dispatch_count / remote_dispatch_total) * 100, 1)
                if remote_dispatch_total > 0 and str(node.get("role") or "") == "worker" and str(node.get("node_id") or "").strip() != str(config.cluster_node_id or "").strip()
                else 0
            )
            node["dispatch_stats"] = stats

        healthy_nodes = [node for node in nodes if bool(node.get("healthy"))]
        total_capacity = sum(max(0, int(node.get("node_max_concurrency") or 0)) for node in healthy_nodes)
        available_capacity = sum(max(0, int(node.get("available_slots") or 0)) for node in healthy_nodes)
        average_roundtrip = [
            int(node.get("reported_roundtrip_ms"))
            for node in healthy_nodes
            if node.get("reported_roundtrip_ms") is not None
        ]
        return {
            "enabled": self.is_enabled(),
            "role": self.role,
            "node_id": config.cluster_node_id,
            "node_name": config.cluster_node_name,
            "master_base_url": config.cluster_master_base_url,
            "node_public_base_url": config.cluster_effective_node_public_base_url,
            "cluster_key_masked": config.mask_cluster_key(),
            "node_weight": int(config.cluster_node_weight),
            "node_max_concurrency": int(config.cluster_node_max_concurrency),
            "heartbeat_interval_seconds": int(config.cluster_heartbeat_interval_seconds),
            "heartbeat_timeout_seconds": int(config.cluster_heartbeat_timeout_seconds),
            "dispatch_timeout_seconds": int(config.cluster_dispatch_timeout_seconds),
            "prefer_local": bool(config.cluster_prefer_local),
            "total_nodes": len(nodes),
            "healthy_nodes": len(healthy_nodes),
            "effective_capacity": total_capacity,
            "available_capacity": available_capacity,
            "remote_dispatch_total": remote_dispatch_total,
            "remote_dispatch_success": remote_dispatch_success,
            "remote_dispatch_failure": remote_dispatch_failure,
            "average_heartbeat_ms": (
                int(sum(average_roundtrip) / len(average_roundtrip)) if average_roundtrip else None
            ),
            "last_refresh_at": _utc_now_iso(),
            "nodes": nodes,
            "master_sync": dict(self._last_master_sync),
            "delegated_auto_login_jobs": await self.list_delegated_auto_login_jobs(),
        }
