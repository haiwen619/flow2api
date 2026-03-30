"""Load balancing module for Flow2API"""
import asyncio
import random
from typing import Optional, Dict
from ..core.models import Token
from ..core.config import config
from ..core.account_tiers import (
    get_paygate_tier_label,
    get_required_paygate_tier_for_model,
    normalize_user_paygate_tier,
    supports_model_for_tier,
)
from .concurrency_manager import ConcurrencyManager
from ..core.logger import debug_logger


class LoadBalancer:
    """Token load balancer with load-aware selection"""

    def __init__(self, token_manager, concurrency_manager: Optional[ConcurrencyManager] = None):
        self.token_manager = token_manager
        self.concurrency_manager = concurrency_manager
        self.cluster_manager = None
        self._image_pending: Dict[int, int] = {}
        self._video_pending: Dict[int, int] = {}
        self._pending_lock = asyncio.Lock()
        self._round_robin_state: Dict[str, Optional[int]] = {"image": None, "video": None, "default": None}
        self._rr_lock = asyncio.Lock()

    def set_cluster_manager(self, cluster_manager):
        self.cluster_manager = cluster_manager

    async def _get_pending_count(self, token_id: int, for_image_generation: bool, for_video_generation: bool) -> int:
        async with self._pending_lock:
            if for_image_generation:
                return max(0, int(self._image_pending.get(token_id, 0)))
            if for_video_generation:
                return max(0, int(self._video_pending.get(token_id, 0)))
            return 0

    async def _add_pending(self, token_id: int, for_image_generation: bool, for_video_generation: bool):
        async with self._pending_lock:
            if for_image_generation:
                self._image_pending[token_id] = max(0, int(self._image_pending.get(token_id, 0))) + 1
            elif for_video_generation:
                self._video_pending[token_id] = max(0, int(self._video_pending.get(token_id, 0))) + 1

    async def release_pending(self, token_id: int, for_image_generation: bool = False, for_video_generation: bool = False):
        async with self._pending_lock:
            if for_image_generation:
                current = max(0, int(self._image_pending.get(token_id, 0)))
                if current <= 1:
                    self._image_pending.pop(token_id, None)
                else:
                    self._image_pending[token_id] = current - 1
            elif for_video_generation:
                current = max(0, int(self._video_pending.get(token_id, 0)))
                if current <= 1:
                    self._video_pending.pop(token_id, None)
                else:
                    self._video_pending[token_id] = current - 1

    async def get_all_load_status(self) -> dict:
        """返回合并硬并发与 pending 的实时负载快照。"""
        if self.concurrency_manager:
            base_status = await self.concurrency_manager.get_all_concurrency_status()
        else:
            base_status = {
                "tokens": {},
                "summary": {
                    "total_image_inflight": 0,
                    "total_video_inflight": 0,
                    "total_image_capacity": None,
                    "total_video_capacity": None,
                    "token_count": 0,
                    "image_idle": 0,
                    "image_busy": 0,
                    "image_saturated": 0,
                    "video_idle": 0,
                    "video_busy": 0,
                    "video_saturated": 0,
                },
            }

        async with self._pending_lock:
            image_pending = {int(k): max(0, int(v)) for k, v in self._image_pending.items()}
            video_pending = {int(k): max(0, int(v)) for k, v in self._video_pending.items()}

        raw_tokens = base_status.get("tokens") or {}
        base_summary = base_status.get("summary") or {}
        all_token_ids = {
            int(tid) for tid in raw_tokens.keys()
        } | set(image_pending.keys()) | set(video_pending.keys())

        tokens_status = {}
        total_image_inflight = 0
        total_video_inflight = 0
        image_idle = 0
        image_busy = 0
        image_saturated = 0
        video_idle = 0
        video_busy = 0
        video_saturated = 0

        for tid in all_token_ids:
            token_status = raw_tokens.get(tid) or raw_tokens.get(str(tid)) or {}
            hard_img_inflight = max(0, int(token_status.get("image_inflight") or 0))
            hard_vid_inflight = max(0, int(token_status.get("video_inflight") or 0))
            img_pending = image_pending.get(tid, 0)
            vid_pending = video_pending.get(tid, 0)
            img_limit = token_status.get("image_limit")
            vid_limit = token_status.get("video_limit")
            effective_img_inflight = hard_img_inflight + img_pending
            effective_vid_inflight = hard_vid_inflight + vid_pending
            img_remaining = max(0, img_limit - effective_img_inflight) if img_limit is not None else None
            vid_remaining = max(0, vid_limit - effective_vid_inflight) if vid_limit is not None else None

            tokens_status[tid] = {
                "image_inflight": effective_img_inflight,
                "image_hard_inflight": hard_img_inflight,
                "image_pending": img_pending,
                "image_limit": img_limit,
                "image_remaining": img_remaining,
                "video_inflight": effective_vid_inflight,
                "video_hard_inflight": hard_vid_inflight,
                "video_pending": vid_pending,
                "video_limit": vid_limit,
                "video_remaining": vid_remaining,
            }

            total_image_inflight += effective_img_inflight
            total_video_inflight += effective_vid_inflight

            if effective_img_inflight == 0:
                image_idle += 1
            elif img_limit is not None and effective_img_inflight >= img_limit:
                image_saturated += 1
            else:
                image_busy += 1

            if effective_vid_inflight == 0:
                video_idle += 1
            elif vid_limit is not None and effective_vid_inflight >= vid_limit:
                video_saturated += 1
            else:
                video_busy += 1

        return {
            "tokens": tokens_status,
            "summary": {
                "total_image_inflight": total_image_inflight,
                "total_video_inflight": total_video_inflight,
                "total_image_hard_inflight": int(base_summary.get("total_image_inflight") or 0),
                "total_video_hard_inflight": int(base_summary.get("total_video_inflight") or 0),
                "total_image_pending": sum(image_pending.values()),
                "total_video_pending": sum(video_pending.values()),
                "total_image_capacity": base_summary.get("total_image_capacity"),
                "total_video_capacity": base_summary.get("total_video_capacity"),
                "token_count": len(all_token_ids),
                "image_idle": image_idle,
                "image_busy": image_busy,
                "image_saturated": image_saturated,
                "video_idle": video_idle,
                "video_busy": video_busy,
                "video_saturated": video_saturated,
            },
        }

    async def _get_token_load(self, token_id: int, for_image_generation: bool, for_video_generation: bool) -> tuple[int, Optional[int]]:
        """获取 token 当前负载。

        Returns:
            (inflight, remaining)
            remaining 为 None 表示无限制
        """
        if not self.concurrency_manager:
            return 0, None

        if for_image_generation:
            inflight = await self.concurrency_manager.get_image_inflight(token_id)
            remaining = await self.concurrency_manager.get_image_remaining(token_id)
            pending = await self._get_pending_count(token_id, True, False)
            effective_inflight = inflight + pending
            if remaining is not None:
                remaining = max(0, remaining - pending)
            return effective_inflight, remaining

        if for_video_generation:
            inflight = await self.concurrency_manager.get_video_inflight(token_id)
            remaining = await self.concurrency_manager.get_video_remaining(token_id)
            pending = await self._get_pending_count(token_id, False, True)
            effective_inflight = inflight + pending
            if remaining is not None:
                remaining = max(0, remaining - pending)
            return effective_inflight, remaining

        return 0, None

    async def _reserve_slot(self, token_id: int, for_image_generation: bool, for_video_generation: bool) -> bool:
        """尝试为当前 token 预占一个生成槽位。"""
        if not self.concurrency_manager:
            return True

        if for_image_generation:
            return await self.concurrency_manager.acquire_image(token_id)

        if for_video_generation:
            return await self.concurrency_manager.acquire_video(token_id)

        return True

    async def _select_round_robin(self, tokens: list[dict], scenario: str) -> Optional[dict]:
        """Select candidate in round-robin order for the given scenario."""
        if not tokens:
            return None

        tokens_sorted = sorted(tokens, key=lambda item: item["token"].id or 0)
        async with self._rr_lock:
            last_id = self._round_robin_state.get(scenario)
            start_idx = 0
            if last_id is not None:
                for idx, item in enumerate(tokens_sorted):
                    if item["token"].id == last_id:
                        start_idx = (idx + 1) % len(tokens_sorted)
                        break
            selected = tokens_sorted[start_idx]
            self._round_robin_state[scenario] = selected["token"].id
        return selected

    async def select_token(
        self,
        for_image_generation: bool = False,
        for_video_generation: bool = False,
        model: Optional[str] = None,
        reserve: bool = False,
        enforce_concurrency_filter: bool = True,
        track_pending: bool = False,
    ) -> Optional[Token]:
        """
        Select a token using load-aware balancing

        Args:
            for_image_generation: If True, only select tokens with image_enabled=True
            for_video_generation: If True, only select tokens with video_enabled=True
            model: Model name (used to filter tokens for specific models)
            reserve: Whether to atomically reserve one concurrency slot for the selected token
            enforce_concurrency_filter:
                Whether to pre-filter tokens by current inflight/remaining capacity.
                For reserve=False generation paths, this should usually be False so
                requests can enter the downstream wait queue instead of failing fast.
            track_pending:
                Whether to count the selected token as a queued request immediately.
                This smooths burst distribution before the hard concurrency slot is acquired.

        Returns:
            Selected token or None if no available tokens
        """
        debug_logger.log_info(
            f"[LOAD_BALANCER] 开始选择Token (图片生成={for_image_generation}, "
            f"视频生成={for_video_generation}, 模型={model}, 预占槽位={reserve})"
        )

        active_tokens = await self.token_manager.get_active_tokens()
        debug_logger.log_info(f"[LOAD_BALANCER] 获取到 {len(active_tokens)} 个活跃Token")

        worker_used_emails = set()
        if self.cluster_manager:
            try:
                if self.cluster_manager.is_master():
                    # 主节点：实时聚合所有子节点 + 自身占用的 email
                    worker_used_emails = await self.cluster_manager._build_global_occupied_emails()
                    worker_used_emails = set(worker_used_emails)
                else:
                    # 子节点：使用上次 heartbeat 响应中缓存的全局占用 email 列表
                    worker_used_emails = self.cluster_manager.get_globally_occupied_emails()
            except Exception as exc:
                debug_logger.log_warning(f"[LOAD_BALANCER] 获取全局 Token 占用状态失败: {exc}")

        if not active_tokens:
            debug_logger.log_info(f"[LOAD_BALANCER] ❌ 没有活跃的Token")
            return None

        available_tokens = []
        filtered_reasons = {}
        required_tier = get_required_paygate_tier_for_model(model)

        for token in active_tokens:
            token_email = str(getattr(token, "email", "") or "").strip().lower()
            if token_email and token_email in worker_used_emails:
                filtered_reasons[token.id] = "已被其他节点占用"
                continue
            normalized_tier = normalize_user_paygate_tier(token.user_paygate_tier)
            if model and not supports_model_for_tier(model, normalized_tier):
                filtered_reasons[token.id] = '账号等级不足，需要 ' + get_paygate_tier_label(required_tier)
                continue
            if for_image_generation:
                if not token.image_enabled:
                    filtered_reasons[token.id] = "图片生成已禁用"
                    continue

                if (
                    enforce_concurrency_filter
                    and self.concurrency_manager
                    and not await self.concurrency_manager.can_use_image(token.id)
                ):
                    filtered_reasons[token.id] = "图片并发已满"
                    continue

            if for_video_generation:
                if not token.video_enabled:
                    filtered_reasons[token.id] = "视频生成已禁用"
                    continue

                if (
                    enforce_concurrency_filter
                    and self.concurrency_manager
                    and not await self.concurrency_manager.can_use_video(token.id)
                ):
                    filtered_reasons[token.id] = "视频并发已满"
                    continue

            inflight, remaining = await self._get_token_load(
                token.id,
                for_image_generation=for_image_generation,
                for_video_generation=for_video_generation
            )
            available_tokens.append({
                "token": token,
                "inflight": inflight,
                "remaining": remaining,
                "needs_refresh": self.token_manager.needs_at_refresh(token),
                "random": random.random()
            })

        if filtered_reasons:
            debug_logger.log_info(f"[LOAD_BALANCER] 已过滤Token:")
            for token_id, reason in filtered_reasons.items():
                debug_logger.log_info(f"[LOAD_BALANCER]   - Token {token_id}: {reason}")

        if not available_tokens:
            debug_logger.log_info(f"[LOAD_BALANCER] ❌ 没有可用的Token (图片生成={for_image_generation}, 视频生成={for_video_generation})")
            return None

        # 最低 in-flight 优先；有并发上限时，剩余槽位更多的 token 优先；最后随机打散
        call_mode = config.call_logic_mode
        if call_mode == "polling":
            scenario = "default"
            if for_image_generation:
                scenario = "image"
            elif for_video_generation:
                scenario = "video"

            ordered_candidates = []
            first_candidate = await self._select_round_robin(available_tokens, scenario)
            if first_candidate is not None:
                ordered_candidates.append(first_candidate)
                ordered_candidates.extend(
                    item for item in sorted(available_tokens, key=lambda item: item["token"].id or 0)
                    if item["token"].id != first_candidate["token"].id
                )
            available_tokens = ordered_candidates
        else:
            available_tokens.sort(
                key=lambda item: (
                    1 if item["needs_refresh"] else 0,
                    item["inflight"],
                    0 if item["remaining"] is None else 1,
                    -(item["remaining"] or 0),
                    item["random"]
                )
            )

        ready_candidates = [item for item in available_tokens if not item["needs_refresh"]]
        refresh_candidates = [item for item in available_tokens if item["needs_refresh"]]
        if ready_candidates and refresh_candidates:
            available_tokens = ready_candidates + refresh_candidates

        debug_logger.log_info("[LOAD_BALANCER] 候选Token负载:")
        for item in available_tokens:
            token = item["token"]
            remaining = "unlimited" if item["remaining"] is None else item["remaining"]
            debug_logger.log_info(
                f"[LOAD_BALANCER]   - Token {token.id} ({token.email}) "
                f"inflight={item['inflight']}, remaining={remaining}, "
                f"needs_refresh={item['needs_refresh']}, credits={token.credits}"
            )

        # 只为候选列表中真正尝试到的 token 做 AT 校验，避免每次请求把所有 token 全扫一遍
        for item in available_tokens:
            token = item["token"]
            token_id = token.id

            token = await self.token_manager.ensure_valid_token(token)
            if not token:
                debug_logger.log_info(f"[LOAD_BALANCER] 跳过 Token {token_id}: AT无效或已过期")
                continue

            if reserve and not await self._reserve_slot(token.id, for_image_generation, for_video_generation):
                debug_logger.log_info(f"[LOAD_BALANCER] 跳过 Token {token.id}: 预占槽位失败")
                continue

            if track_pending and not reserve:
                await self._add_pending(token.id, for_image_generation, for_video_generation)

            debug_logger.log_info(
                f"[LOAD_BALANCER] ✅ 已选择Token {token.id} ({token.email}) - "
                f"余额: {token.credits}, inflight={item['inflight']}"
            )
            return token

        debug_logger.log_info(f"[LOAD_BALANCER] ❌ 候选Token均不可用 (图片生成={for_image_generation}, 视频生成={for_video_generation})")
        return None

    async def get_unavailable_reason(
        self,
        *,
        for_image_generation: bool = False,
        for_video_generation: bool = False,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """给出更明确的“无可用账号”原因，优先用于分辨率/tier 档位提示。"""
        active_tokens = await self.token_manager.get_active_tokens()
        if not active_tokens:
            return None

        required_tier = get_required_paygate_tier_for_model(model)
        supported_tokens = []
        for token in active_tokens:
            normalized_tier = normalize_user_paygate_tier(token.user_paygate_tier)
            if model and not supports_model_for_tier(model, normalized_tier):
                continue
            supported_tokens.append(token)

        if model and not supported_tokens:
            tier_label = get_paygate_tier_label(required_tier)
            return f"当前模型需要 {tier_label} 账号，但没有可用的 {tier_label} 账号: {model}"

        capability_tokens = []
        for token in supported_tokens:
            if for_image_generation and not token.image_enabled:
                continue
            if for_video_generation and not token.video_enabled:
                continue
            capability_tokens.append(token)

        if supported_tokens and not capability_tokens:
            if for_image_generation:
                return "当前有符合档位的账号，但图片生成功能已全部禁用。"
            if for_video_generation:
                return "当前有符合档位的账号，但视频生成功能已全部禁用。"

        return None
