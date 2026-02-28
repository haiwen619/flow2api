"""Token manager for Flow2API with AT auto-refresh"""
import asyncio
import base64
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from ..core.database import Database
from ..core.models import Token, Project
from ..core.logger import debug_logger
from .flow_client import FlowClient
from .proxy_manager import ProxyManager


class TokenManager:
    """Token lifecycle manager with AT auto-refresh"""

    def __init__(self, db: Database, flow_client: FlowClient):
        self.db = db
        self.flow_client = flow_client
        self._lock = asyncio.Lock()

    # ========== Token CRUD ==========

    async def get_all_tokens(self) -> List[Token]:
        """Get all tokens"""
        return await self.db.get_all_tokens()

    async def get_active_tokens(self) -> List[Token]:
        """Get all active tokens"""
        return await self.db.get_active_tokens()

    async def get_token(self, token_id: int) -> Optional[Token]:
        """Get token by ID"""
        return await self.db.get_token(token_id)

    async def delete_token(self, token_id: int):
        """Delete token"""
        await self.db.delete_token(token_id)

    async def enable_token(self, token_id: int):
        """Enable a token and reset error count"""
        # Enable the token
        await self.db.update_token(token_id, is_active=True)
        # Reset error count when enabling (only reset total error_count, keep today_error_count)
        await self.db.reset_error_count(token_id)

    async def disable_token(self, token_id: int):
        """Disable a token"""
        await self.db.update_token(token_id, is_active=False)

    # ========== Token添加 (支持Project创建) ==========

    async def add_token(
        self,
        st: str,
        cookie: Optional[str] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        remark: Optional[str] = None,
        image_enabled: bool = True,
        video_enabled: bool = True,
        image_concurrency: int = -1,
        video_concurrency: int = -1
    ) -> Token:
        """Add a new token

        Args:
            st: Session Token (必需)
            cookie: 完整 Cookie Header（可选）
            project_id: 项目ID (可选,如果提供则直接使用,不创建新项目)
            project_name: 项目名称 (可选,如果不提供则自动生成)
            remark: 备注
            image_enabled: 是否启用图片生成
            video_enabled: 是否启用视频生成
            image_concurrency: 图片并发限制
            video_concurrency: 视频并发限制

        Returns:
            Token object
        """
        # Step 1: 检查ST是否已存在
        existing_token = await self.db.get_token_by_st(st)
        if existing_token:
            raise ValueError(f"Token 已存在（邮箱: {existing_token.email}）")

        # Step 2: 使用ST转换AT
        debug_logger.log_info(f"[ADD_TOKEN] Converting ST to AT...")
        try:
            result = await self.flow_client.st_to_at(st)
            at = result["access_token"]
            expires = result.get("expires")
            user_info = result.get("user", {})
            email = user_info.get("email", "")
            name = user_info.get("name", email.split("@")[0] if email else "")

            # 解析过期时间
            at_expires = None
            if expires:
                try:
                    at_expires = datetime.fromisoformat(expires.replace('Z', '+00:00'))
                except:
                    pass

        except Exception as e:
            raise ValueError(f"ST转AT失败: {str(e)}")

        # Step 3: 查询余额
        try:
            credits_result = await self.flow_client.get_credits(at)
            credits = credits_result.get("credits", 0)
            user_paygate_tier = credits_result.get("userPaygateTier")
        except:
            credits = 0
            user_paygate_tier = None

        # Step 4: 处理Project ID和名称
        if project_id:
            # 用户提供了project_id,直接使用
            debug_logger.log_info(f"[ADD_TOKEN] Using provided project_id: {project_id}")
            if not project_name:
                # 如果没有提供project_name,生成一个
                now = datetime.now()
                project_name = now.strftime("%b %d - %H:%M")
        else:
            # 用户没有提供project_id,需要创建新项目
            if not project_name:
                # 自动生成项目名称
                now = datetime.now()
                project_name = now.strftime("%b %d - %H:%M")

            try:
                project_id = await self.flow_client.create_project(st, project_name)
                debug_logger.log_info(f"[ADD_TOKEN] Created new project: {project_name} (ID: {project_id})")
            except Exception as e:
                raise ValueError(f"创建项目失败: {str(e)}")

        # Step 5: 创建Token对象
        token = Token(
            st=st,
            cookie=cookie,
            at=at,
            at_expires=at_expires,
            email=email,
            name=name,
            remark=remark,
            is_active=True,
            credits=credits,
            user_paygate_tier=user_paygate_tier,
            current_project_id=project_id,
            current_project_name=project_name,
            image_enabled=image_enabled,
            video_enabled=video_enabled,
            image_concurrency=image_concurrency,
            video_concurrency=video_concurrency
        )

        # Step 6: 保存到数据库
        token_id = await self.db.add_token(token)
        token.id = token_id

        # Step 7: 保存Project到数据库
        project = Project(
            project_id=project_id,
            token_id=token_id,
            project_name=project_name,
            tool_name="PINHOLE"
        )
        await self.db.add_project(project)

        debug_logger.log_info(f"[ADD_TOKEN] Token added successfully (ID: {token_id}, Email: {email})")
        return token

    async def update_token(
        self,
        token_id: int,
        st: Optional[str] = None,
        cookie: Optional[str] = None,
        at: Optional[str] = None,
        at_expires: Optional[datetime] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        remark: Optional[str] = None,
        image_enabled: Optional[bool] = None,
        video_enabled: Optional[bool] = None,
        image_concurrency: Optional[int] = None,
        video_concurrency: Optional[int] = None
    ):
        """Update token (支持修改project_id和project_name)

        当用户编辑保存token时，如果token未过期，自动清空429禁用状态
        """
        update_fields = {}

        if st is not None:
            update_fields["st"] = st
        if cookie is not None:
            update_fields["cookie"] = cookie
        if at is not None:
            update_fields["at"] = at
        if at_expires is not None:
            update_fields["at_expires"] = at_expires
        if project_id is not None:
            update_fields["current_project_id"] = project_id
        if project_name is not None:
            update_fields["current_project_name"] = project_name
        if remark is not None:
            update_fields["remark"] = remark
        if image_enabled is not None:
            update_fields["image_enabled"] = image_enabled
        if video_enabled is not None:
            update_fields["video_enabled"] = video_enabled
        if image_concurrency is not None:
            update_fields["image_concurrency"] = image_concurrency
        if video_concurrency is not None:
            update_fields["video_concurrency"] = video_concurrency

        # 检查token是否因429被禁用，如果是且未过期，则清空429状态
        token = await self.db.get_token(token_id)
        if token and token.ban_reason == "429_rate_limit":
            # 检查token是否过期
            is_expired = False
            if token.at_expires:
                now = datetime.now(timezone.utc)
                if token.at_expires.tzinfo is None:
                    at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
                else:
                    at_expires_aware = token.at_expires
                is_expired = at_expires_aware <= now

            # 如果未过期，清空429禁用状态
            if not is_expired:
                debug_logger.log_info(f"[UPDATE_TOKEN] Token {token_id} 编辑保存，清空429禁用状态")
                update_fields["ban_reason"] = None
                update_fields["banned_at"] = None

        if update_fields:
            await self.db.update_token(token_id, **update_fields)

    # ========== AT自动刷新逻辑 (核心) ==========

    async def is_at_valid(self, token_id: int) -> bool:
        """检查AT是否有效,如果无效或即将过期则自动刷新

        Returns:
            True if AT is valid or refreshed successfully
            False if AT cannot be refreshed
        """
        token = await self.db.get_token(token_id)
        if not token:
            return False

        # 如果AT不存在,需要刷新
        if not token.at:
            debug_logger.log_info(f"[AT_CHECK] Token {token_id}: AT不存在,需要刷新")
            return await self._refresh_at(token_id)

        # 如果没有过期时间,假设需要刷新
        if not token.at_expires:
            debug_logger.log_info(f"[AT_CHECK] Token {token_id}: AT过期时间未知,尝试刷新")
            return await self._refresh_at(token_id)

        # 检查是否即将过期 (提前1小时刷新)
        now = datetime.now(timezone.utc)
        # 确保at_expires也是timezone-aware
        if token.at_expires.tzinfo is None:
            at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
        else:
            at_expires_aware = token.at_expires

        time_until_expiry = at_expires_aware - now

        if time_until_expiry.total_seconds() < 3600:  # 1 hour (3600 seconds)
            debug_logger.log_info(f"[AT_CHECK] Token {token_id}: AT即将过期 (剩余 {time_until_expiry.total_seconds():.0f} 秒),需要刷新")
            return await self._refresh_at(token_id)

        # AT有效
        return True


    async def _refresh_at(self, token_id: int) -> bool:
        """内部方法: 刷新AT

        如果 AT 刷新失败（ST 可能过期），会尝试通过浏览器自动刷新 ST，
        然后重试 AT 刷新。

        Returns:
            True if refresh successful, False otherwise
        """
        async with self._lock:
            token = await self.db.get_token(token_id)
            if not token:
                return False

            # # 第一次尝试刷新 AT
            # result = await self._do_refresh_at(token_id, token.st)
            # if result:
            #     return True

            # AT 刷新失败，可选尝试 HTTP reAuth 恢复可用 AT（可能先拿到 ST 再换 AT，或直接拿到可用 AT）
            from ..core.config import config
            if config.enable_reauth_refresh:
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 第一次 AT 刷新失败，尝试 reAuth 恢复 AT...")
                reauth_result = await self._try_refresh_at_via_reauth(token_id, token)
                if reauth_result:
                    debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: reAuth 恢复 AT 成功")
                    return True
            else:
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: reAuth 恢复已禁用（flow.enable_reauth_refresh=false），跳过")

            # 继续尝试 personal 模式浏览器刷新 ST
            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 尝试浏览器刷新 ST...")
            new_st = await self._try_refresh_st(token_id, token)
            if new_st:
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 浏览器刷新 ST 成功，重试 AT 刷新...")
                result = await self._do_refresh_at(token_id, new_st)
                if result:
                    return True

            # 所有刷新尝试都失败，禁用 Token
            debug_logger.log_error(f"[AT_REFRESH] Token {token_id}: 所有刷新尝试失败，禁用 Token")
            await self.disable_token(token_id)
            return False

    async def refresh_cookie_via_reauth(self, token_id: int) -> bool:
        """仅通过 reAuth 刷新 Cookie/会话，不执行首次 ST->AT 直刷。

        用于手动“刷新Cookie”按钮：
        - 直接走 HTTP reAuth
        - 跳过 `_refresh_at` 中第一次 `_do_refresh_at` 尝试
        """
        async with self._lock:
            token = await self.db.get_token(token_id)
            if not token:
                return False

            old_cookie = str(token.cookie or "").strip()
            debug_logger.log_info(f"[REAUTH_ONLY] Token {token_id}: 开始仅 reAuth 刷新 Cookie...")
            success = await self._try_refresh_at_via_reauth(token_id, token)
            if not success:
                debug_logger.log_warning(f"[REAUTH_ONLY] Token {token_id}: reAuth 刷新失败")
                return False

            updated = await self.db.get_token(token_id)
            new_cookie = str((updated.cookie if updated else "") or "").strip()
            cookie_changed = bool(new_cookie and new_cookie != old_cookie)
            debug_logger.log_info(
                f"[REAUTH_ONLY] Token {token_id}: reAuth 刷新成功 cookie_changed={cookie_changed}"
            )
            return True

    async def _do_refresh_at(self, token_id: int, st: str, st_raw: Optional[str] = None) -> bool:
        """执行 AT 刷新的核心逻辑

        Args:
            token_id: Token ID
            st: Session Token

        Returns:
            True if refresh successful AND AT is valid, False otherwise
        """
        try:
            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 开始刷新AT...")

            # 兼容输入：若误传入 Cookie/Set-Cookie 文本（含 Path/Expires 等属性），先提取纯 ST。
            st_input = str(st or "")
            if "__Secure-next-auth.session-token" in st_input or ";" in st_input:
                try:
                    from reAuth.refresh_cookie_before_relogin import extract_session_token_from_cookie_header
                    extracted = extract_session_token_from_cookie_header(st_input)
                    if extracted:
                        debug_logger.log_info(
                            f"[AT_REFRESH] Token {token_id}: 检测到Cookie/Set-Cookie格式输入，已提取纯ST（输入长度={len(st_input)} -> ST长度={len(extracted)}）"
                        )
                        st = extracted
                except Exception as norm_err:
                    debug_logger.log_warning(
                        f"[AT_REFRESH] Token {token_id}: ST 输入规范化失败，继续使用原值: {norm_err}"
                    )
            
            # 临时调试位：如需强制使用网页最新 ST 进行测试，可在此处填入。
            # 留空字符串时不生效，继续使用函数入参 st。
            debug_st_override = ""
            st_for_refresh = (str(debug_st_override).strip() or st)
            if str(debug_st_override).strip():
                debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: 使用临时调试 ST 覆盖入参 ST")
            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: 当前ST长度={len(str(st_for_refresh or ''))}")
            if st_raw:
                print(f"  - 当前 ST(原文): {st_raw}")
            print(f"  - 当前 ST: {st_for_refresh}")
            # 使用ST转AT
            result = await self.flow_client.st_to_at(st_for_refresh)
            new_at = result["access_token"]
            expires = result.get("expires")

            # 解析过期时间
            new_at_expires = None
            if expires:
                try:
                    new_at_expires = datetime.fromisoformat(expires.replace('Z', '+00:00'))
                except:
                    pass

            # 更新数据库
            await self.db.update_token(
                token_id,
                at=new_at,
                at_expires=new_at_expires
            )

            debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: AT刷新成功")
            debug_logger.log_info(f"  - 新过期时间(UTC): {new_at_expires}")
            beijing_expires_text = None
            if new_at_expires is not None:
                try:
                    beijing_tz = timezone(timedelta(hours=8))
                    if new_at_expires.tzinfo is None:
                        exp_aware = new_at_expires.replace(tzinfo=timezone.utc)
                    else:
                        exp_aware = new_at_expires
                    beijing_expires_text = exp_aware.astimezone(beijing_tz).isoformat()
                except Exception:
                    beijing_expires_text = None
            debug_logger.log_info(f"  - 新过期时间(北京时间): {beijing_expires_text or '未知'}")

            # 验证 AT 有效性：通过 get_credits 测试
            try:
                print(f"  - 验证新 AT 是否有效...")
                print(f"  - 新 AT: {new_at}")
                credits_result = await self.flow_client.get_credits(new_at)
                await self.db.update_token(
                    token_id,
                    credits=credits_result.get("credits", 0)
                )
                debug_logger.log_info(f"[AT_REFRESH] Token {token_id}: AT 验证成功（余额: {credits_result.get('credits', 0)}）")
                return True
            except Exception as verify_err:
                # AT 验证失败（可能返回 401），说明 ST 已过期
                error_msg = str(verify_err)
                if "401" in error_msg or "UNAUTHENTICATED" in error_msg:
                    debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: AT 验证失败 (401)，ST 可能已过期")
                    return False
                else:
                    # 其他错误（如网络问题），仍视为成功
                    debug_logger.log_warning(f"[AT_REFRESH] Token {token_id}: AT 验证时发生非认证错误: {error_msg}")
                    return True

        except Exception as e:
            debug_logger.log_error(f"[AT_REFRESH] Token {token_id}: AT刷新失败 - {str(e)}")
            return False

    def _parse_jwt_exp(self, token: str) -> Optional[datetime]:
        """从 JWT 中提取 exp 并转为 UTC datetime"""
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload = parts[1]
            payload += "=" * ((4 - len(payload) % 4) % 4)
            decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
            import json
            payload_obj = json.loads(decoded)
            exp = payload_obj.get("exp")
            if not exp:
                return None
            return datetime.fromtimestamp(int(exp), tz=timezone.utc)
        except Exception:
            return None

    async def _accept_direct_at_if_valid(self, token_id: int, candidate_at: str) -> bool:
        """将候选 token 作为 AT 直接验证并写回"""
        try:
            credits_result = await self.flow_client.get_credits(candidate_at)
            at_expires = self._parse_jwt_exp(candidate_at)
            await self.db.update_token(
                token_id,
                at=candidate_at,
                at_expires=at_expires,
                credits=credits_result.get("credits", 0),
            )
            debug_logger.log_info(
                f"[REAUTH_AT] Token {token_id}: 候选 token 可作为 AT 使用（余额: {credits_result.get('credits', 0)}）"
            )
            return True
        except Exception as e:
            debug_logger.log_warning(f"[REAUTH_AT] Token {token_id}: 候选 token 作为 AT 验证失败 - {str(e)}")
            return False

    async def _try_refresh_at_via_reauth(self, token_id: int, token) -> bool:
        """尝试通过 reAuth HTTP 流程恢复 AT（优先按 ST/RT->AT，兼容 AT 直验）"""
        try:
            if not token.current_project_id:
                debug_logger.log_warning(f"[REAUTH_ST] Token {token_id} 没有 project_id，无法执行 reAuth")
                return False

            debug_logger.log_info(f"[REAUTH_AT] Token {token_id}: 开始执行 reAuth，尝试恢复 AT...")

            # 延迟导入，避免主流程无谓依赖
            from reAuth.refresh_cookie_before_relogin import (
                ReAuthAccount,
                refresh_cookie_before_relogin,
                _st_to_at_via_auth_session,
                _verify_at_via_get_credits,
            )

            # 优先使用完整 Cookie（由账号池自动化同步），没有则回退最小 ST Cookie
            initial_cookie_header = str(token.cookie or "").strip() or f"__Secure-next-auth.session-token={token.st}"
            old_st_value = str(token.st or "").strip()

            proxy_url = None
            try:
                proxy_config = await self.db.get_proxy_config()
                if proxy_config and proxy_config.enabled and proxy_config.proxy_url:
                    proxy_url = proxy_config.proxy_url
            except Exception as proxy_err:
                debug_logger.log_warning(f"[REAUTH_AT] Token {token_id}: 读取代理配置失败，继续直连: {proxy_err}")

            seed_cookie = initial_cookie_header
            last_candidate_token = ""
            candidate_source = "__Secure-next-auth.session-token"
            # 暂时收敛为单次纯协议尝试，便于聚焦六步链路调试。
            strategies = [
                ("default-pass1", False, False),
            ]

            for attempt_idx, (strategy_name, strict_mode, use_cookie_file) in enumerate(strategies, start=1):
                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: 协议刷新尝试 {attempt_idx}/{len(strategies)} strategy={strategy_name} strict={strict_mode}"
                )

                def _run_reauth_once(
                    cookie_seed: str = seed_cookie,
                    strict_value: bool = strict_mode,
                    use_cookie_file_value: bool = use_cookie_file,
                ) -> tuple[str, str, str]:
                    account = ReAuthAccount(
                        project_id=token.current_project_id,
                        cookie=cookie_seed,
                        cookie_file=(cookie_seed if use_cookie_file_value else None),
                    )
                    refresh_cookie_before_relogin(
                        account=account,
                        timeout=30,
                        proxy_url=proxy_url,
                        strict_java_flow=strict_value,
                    )
                    # 使用 account 内部最终状态，避免依赖 refresh 函数返回值格式。
                    refreshed_cookie_value = str(getattr(account, "cookie", "") or "").strip()
                    refreshed_st_value = str(getattr(account, "final_session_token", "") or "").strip()
                    refreshed_st_raw = str(getattr(account, "final_session_set_cookie_raw", "") or "").strip()
                    return refreshed_cookie_value, refreshed_st_value, refreshed_st_raw

                refreshed_cookie, refreshed_st, refreshed_st_raw = await asyncio.to_thread(_run_reauth_once)
                print(f"  - reAuth[{strategy_name}] 返回 cookie: {refreshed_cookie}")
                print(f"  - reAuth[{strategy_name}] 提取 ST: {refreshed_st_raw or refreshed_st}")
                candidate_token = refreshed_st
                if candidate_token:
                    last_candidate_token = candidate_token

                if refreshed_st_raw and ";" in refreshed_st_raw:
                    debug_logger.log_info(
                        f"[REAUTH_AT] Token {token_id}: 提取到Set-Cookie原文（含Path/Expires属性），ST->AT将使用纯ST值"
                    )
                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: reAuth返回cookie长度={len(str(refreshed_cookie or ''))}, 提取ST长度={len(str(refreshed_st or ''))}, 提取ST原文长度={len(str(refreshed_st_raw or ''))}"
                )
                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: reAuth前后ST对比 old_len={len(old_st_value)} new_len={len(str(refreshed_st or ''))} changed={bool(refreshed_st and refreshed_st != old_st_value)}"
                )

                if refreshed_cookie:
                    # 保存 reAuth 返回的完整 cookie，供下次 reAuth/排障复用
                    await self.db.update_token(token_id, cookie=refreshed_cookie)
                    seed_cookie = refreshed_cookie

                if not candidate_token:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 未提取到可用 session token，继续下一策略"
                    )
                    continue

                debug_logger.log_info(
                    f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 候选 token 来源={candidate_source}"
                )

                if refreshed_st and refreshed_st != old_st_value:
                    await self.db.update_token(token_id, st=refreshed_st)
                    debug_logger.log_info(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 已更新 ST，尝试 ST->AT"
                    )
                    old_st_value = refreshed_st
                elif refreshed_st:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 提取到的 ST 与旧值相同，继续尝试恢复 AT"
                    )
                else:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} 未提取到 ST，仅使用候选 token 继续恢复 AT"
                    )

                # 当前策略下优先使用提取到的 ST，未提取到时回退候选 token。
                fallback_st = refreshed_st or candidate_token

                # 路径0：先用 reAuth 同代理链路直接验证一次（避免 reAuth 与 flow_client 请求画像差异）。
                try:
                    st_to_at_direct = _st_to_at_via_auth_session(
                        fallback_st,
                        proxy_url=proxy_url,
                        connect_timeout=10,
                        timeout=30,
                    )
                    if st_to_at_direct.get("success"):
                        direct_at = str(st_to_at_direct.get("access_token") or "").strip()
                        direct_expires = st_to_at_direct.get("expires")
                        verify_direct = _verify_at_via_get_credits(
                            direct_at,
                            proxy_url=proxy_url,
                            connect_timeout=10,
                            timeout=30,
                        )
                        if verify_direct.get("success"):
                            direct_at_expires = None
                            if isinstance(direct_expires, str) and direct_expires.strip():
                                try:
                                    direct_at_expires = datetime.fromisoformat(direct_expires.replace('Z', '+00:00'))
                                except Exception:
                                    direct_at_expires = None
                            await self.db.update_token(
                                token_id,
                                at=direct_at,
                                at_expires=direct_at_expires,
                                credits=verify_direct.get("credits", 0),
                            )
                            debug_logger.log_info(
                                f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路 AT 验证成功（余额: {verify_direct.get('credits', 0)}）"
                            )
                            return True
                        debug_logger.log_warning(
                            f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路 AT 验证失败: {verify_direct.get('error')}"
                        )
                    else:
                        debug_logger.log_warning(
                            f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路 ST->AT 失败: {st_to_at_direct.get('error')}"
                        )
                except Exception as direct_err:
                    debug_logger.log_warning(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} reAuth同链路验证异常: {direct_err}"
                    )

                # 路径1：继续走现有 flow_client 链路验证。
                if await self._do_refresh_at(
                    token_id,
                    fallback_st,
                    st_raw=(refreshed_st_raw or None),
                ):
                    debug_logger.log_info(
                        f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} ST->AT 成功"
                    )
                    return True

                debug_logger.log_warning(
                    f"[REAUTH_AT] Token {token_id}: strategy={strategy_name} ST->AT 失败，切换下一策略"
                )

            # 路径2（兼容兜底）：极少数环境返回值可能直接是 AT，再尝试按 AT 直验
            if last_candidate_token:
                debug_logger.log_info(f"[REAUTH_AT] Token {token_id}: 所有 ST->AT 策略失败，兼容尝试候选 token 直验 AT...")
                if await self._accept_direct_at_if_valid(token_id, last_candidate_token):
                    return True

            return False

        except Exception as e:
            debug_logger.log_error(f"[REAUTH_AT] Token {token_id}: reAuth 恢复 AT 失败 - {str(e)}")
            return False

    async def _try_refresh_st(self, token_id: int, token) -> Optional[str]:
        """尝试通过浏览器刷新 Session Token

        使用常驻 tab 获取新的 __Secure-next-auth.session-token

        Args:
            token_id: Token ID
            token: Token 对象

        Returns:
            新的 ST 字符串，如果失败返回 None
        """
        try:
            from ..core.config import config

            # 仅在 personal 模式下支持 ST 自动刷新
            if config.captcha_method != "personal":
                debug_logger.log_info(f"[ST_REFRESH] 非 personal 模式，跳过 ST 自动刷新")
                return None

            if not token.current_project_id:
                debug_logger.log_warning(f"[ST_REFRESH] Token {token_id} 没有 project_id，无法刷新 ST")
                return None

            debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: 尝试通过浏览器刷新 ST...")

            from .browser_captcha_personal import BrowserCaptchaService
            service = await BrowserCaptchaService.get_instance(self.db)
            debug_logger.log_info(
                f"[ST_REFRESH] Token {token_id}: BrowserCaptchaService状态 "
                f"initialized={getattr(service, '_initialized', None)} "
                f"running={getattr(service, '_running', None)} "
                f"resident_count={len(getattr(service, '_resident_tabs', {}) or {})}"
            )

            # 强制重置服务实例，避免启动常驻页/残留窗口状态影响本次 ST 刷新。
            try:
                debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: 先执行 BrowserCaptcha 实例重置")
                await service.close()
                debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 实例重置完成")
            except Exception as reset_err:
                debug_logger.log_warning(
                    f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 实例重置异常，继续刷新: {reset_err}"
                )

            # 经验修复：已有常驻标签页（尤其是启动时预热页）会干扰后续 ST 刷新。
            # 在刷新前先统一清理常驻页，等价于“手动先关闭常驻浏览器后再刷新”的成功路径。
            try:
                resident_count_before = service.get_resident_count()
            except Exception:
                resident_count_before = 0
            if resident_count_before > 0:
                debug_logger.log_info(
                    f"[ST_REFRESH] Token {token_id}: 检测到常驻标签页 {resident_count_before} 个，先关闭 BrowserCaptcha 实例再刷新"
                )
                try:
                    await service.close()
                    debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 实例关闭完成")
                except Exception as stop_err:
                    debug_logger.log_warning(
                        f"[ST_REFRESH] Token {token_id}: BrowserCaptcha 关闭异常，继续刷新: {stop_err}"
                    )

            new_st = await service.refresh_session_token(token.current_project_id)
            debug_logger.log_info(
                f"[ST_REFRESH] Token {token_id}: refresh_session_token 返回长度={len(str(new_st or ''))}"
            )
            if new_st and new_st != token.st:
                # 更新数据库中的 ST
                await self.db.update_token(token_id, st=new_st)
                debug_logger.log_info(f"[ST_REFRESH] Token {token_id}: ST 已自动更新")
                return new_st
            elif new_st == token.st:
                debug_logger.log_warning(f"[ST_REFRESH] Token {token_id}: 获取到的 ST 与原 ST 相同，可能登录已失效")
                return None
            else:
                debug_logger.log_warning(f"[ST_REFRESH] Token {token_id}: 无法获取新 ST")
                return None

        except Exception as e:
            debug_logger.log_error(f"[ST_REFRESH] Token {token_id}: 刷新 ST 失败 - {str(e)}")
            debug_logger.log_error(f"[ST_REFRESH] Token {token_id}: 刷新 ST 失败堆栈 - {traceback.format_exc()}")
            return None

    async def ensure_project_exists(self, token_id: int) -> str:
        """确保Token有可用的Project

        Returns:
            project_id
        """
        token = await self.db.get_token(token_id)
        if not token:
            raise ValueError("Token not found")

        # 如果已有project_id,直接返回
        if token.current_project_id:
            return token.current_project_id

        # 创建新Project
        now = datetime.now()
        project_name = now.strftime("%b %d - %H:%M")

        try:
            project_id = await self.flow_client.create_project(token.st, project_name)
            debug_logger.log_info(f"[PROJECT] Created project for token {token_id}: {project_name}")

            # 更新Token
            await self.db.update_token(
                token_id,
                current_project_id=project_id,
                current_project_name=project_name
            )

            # 保存Project到数据库
            project = Project(
                project_id=project_id,
                token_id=token_id,
                project_name=project_name
            )
            await self.db.add_project(project)

            return project_id

        except Exception as e:
            raise ValueError(f"Failed to create project: {str(e)}")

    # ========== Token使用统计 ==========

    async def record_usage(self, token_id: int, is_video: bool = False):
        """Record token usage"""
        await self.db.update_token(token_id, use_count=1, last_used_at=datetime.now())

        if is_video:
            await self.db.increment_token_stats(token_id, "video")
        else:
            await self.db.increment_token_stats(token_id, "image")

    async def record_error(self, token_id: int):
        """Record token error and auto-disable if threshold reached"""
        await self.db.increment_token_stats(token_id, "error")

        # Check if should auto-disable token (based on consecutive errors)
        stats = await self.db.get_token_stats(token_id)
        admin_config = await self.db.get_admin_config()

        if stats and stats.consecutive_error_count >= admin_config.error_ban_threshold:
            debug_logger.log_warning(
                f"[TOKEN_BAN] Token {token_id} consecutive error count ({stats.consecutive_error_count}) "
                f"reached threshold ({admin_config.error_ban_threshold}), auto-disabling"
            )
            await self.disable_token(token_id)

    async def record_success(self, token_id: int):
        """Record successful request (reset consecutive error count)

        This method resets error_count to 0, which is used for auto-disable threshold checking.
        Note: today_error_count and historical statistics are NOT reset.
        """
        await self.db.reset_error_count(token_id)

    async def ban_token_for_429(self, token_id: int):
        """因429错误立即禁用token

        Args:
            token_id: Token ID
        """
        debug_logger.log_warning(f"[429_BAN] 禁用Token {token_id} (原因: 429 Rate Limit)")
        await self.db.update_token(
            token_id,
            is_active=False,
            ban_reason="429_rate_limit",
            banned_at=datetime.now(timezone.utc)
        )

    async def auto_unban_429_tokens(self):
        """自动解禁因429被禁用的token

        规则:
        - 距离禁用时间12小时后自动解禁
        - 仅解禁未过期的token
        - 仅解禁因429被禁用的token
        """
        all_tokens = await self.db.get_all_tokens()
        now = datetime.now(timezone.utc)

        for token in all_tokens:
            # 跳过非429禁用的token
            if token.ban_reason != "429_rate_limit":
                continue

            # 跳过未禁用的token
            if token.is_active:
                continue

            # 跳过没有禁用时间的token
            if not token.banned_at:
                continue

            # 检查token是否已过期
            if token.at_expires:
                # 确保时区一致
                if token.at_expires.tzinfo is None:
                    at_expires_aware = token.at_expires.replace(tzinfo=timezone.utc)
                else:
                    at_expires_aware = token.at_expires

                # 如果已过期，跳过
                if at_expires_aware <= now:
                    debug_logger.log_info(f"[AUTO_UNBAN] Token {token.id} 已过期，跳过解禁")
                    continue

            # 确保banned_at时区一致
            if token.banned_at.tzinfo is None:
                banned_at_aware = token.banned_at.replace(tzinfo=timezone.utc)
            else:
                banned_at_aware = token.banned_at

            # 检查是否已过12小时
            time_since_ban = now - banned_at_aware
            if time_since_ban.total_seconds() >= 12 * 3600:  # 12小时
                debug_logger.log_info(
                    f"[AUTO_UNBAN] 解禁Token {token.id} (禁用时间: {banned_at_aware}, "
                    f"已过 {time_since_ban.total_seconds() / 3600:.1f} 小时)"
                )
                await self.db.update_token(
                    token.id,
                    is_active=True,
                    ban_reason=None,
                    banned_at=None
                )
                # 重置错误计数
                await self.db.reset_error_count(token.id)

    # ========== 余额刷新 ==========

    async def refresh_credits(self, token_id: int) -> int:
        """刷新Token余额

        Returns:
            credits
        """
        token = await self.db.get_token(token_id)
        if not token:
            return 0

        # 确保AT有效
        if not await self.is_at_valid(token_id):
            return 0

        # 重新获取token (AT可能已刷新)
        token = await self.db.get_token(token_id)

        try:
            result = await self.flow_client.get_credits(token.at)
            credits = result.get("credits", 0)

            # 更新数据库
            await self.db.update_token(token_id, credits=credits)

            return credits
        except Exception as e:
            debug_logger.log_error(f"Failed to refresh credits for token {token_id}: {str(e)}")
            return 0
