"""RPA / BrowserAutomation

本模块用途说明（面向 .NET 开发者的中文注释）

概要：
- 本模块用于“账号池一键验活”（Account Pool 验活）场景，通过模拟浏览器流程完成
        Google OAuth 登录并把凭证保存到后端（antigravity 模式）。
- 核心思路：本模块通过 Playwright（或 BitBrowser/CDP）打开目标页面
        `https://labs.google/fx` 并尽力自动完成 Google 登录，最终以是否进入目标页面作为验活结果。

主要流程：
1) 调用后端内部认证逻辑生成 OAuth 授权链接（回调到 localhost 某端口）
2) 使用浏览器打开授权链接，自动完成 Google 登录（尽力处理验证码/2FA），或允许人工介入
3) 等待进入目标页面（labs.google/fx）并确认成功

设计说明（实现细节、可配置点）：
- Playwright：用于在脚本中控制 Chromium（或通过 CDP 连接到 BitBrowser）。
        安装与启动：`pip install playwright` 并运行 `playwright install`（仅使用 Playwright 方式时需要）。
- BitBrowser：项目内使用的本地浏览器管理服务（createBrowser/openBrowser/closeBrowser）。
        本模块优先使用 BitBrowser + CDP 方式连接外部浏览器实例，以降低被 Google 风控的概率。
- 人工介入/调试：模块支持 `manual=True`，在等待进入目标页面阶段允许人工在浏览器中完成交互。
- 日志：使用仓库根的 `log` 模块，所有关键事件会写文件和输出到控制台。

风险与建议：
- Google 会根据环境、UA、行为等触发风控（验证码、手机号验证、设备验证等），
        本模块仅做“best-effort”自动化，碰到挑战页建议切换到可见窗口并人工处理。
- 生产环境推荐使用 BitBrowser 或真实浏览器 profile（避免无头 headless 在 Google 场景下更易触发风控）。

兼容/运行方式：
- 1) 仓库根运行：`python -m Rpa.BrowserAutomation.main`
- 2) 直接进入目录运行：`cd Rpa/BrowserAutomation && python main.py`（运行时会自动加入项目根到 sys.path）

参数/配置说明请查看 `ValidateOptions` 数据类（下方定义），以及文件顶部的 TEST_DEFAULT_* 用于本地测试。
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import random
import re
import sys
import time
import webbrowser
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from urllib import parse as urllib_parse
from urllib import request as urllib_request

# 说明：下面引入的第三方包/模块用途简述（便于来自 .NET 背景的工程师理解）：
# - playwright.async_api: Playwright 的异步 API，用于在 Python 中以编程方式控制浏览器（创建页面、点击、输入等）。
# - playwright_stealth: 可选库，用于注入一些反检测（stealth）脚本以降低被风控识别的概率。
# - Rpa.Login.bit_api: 项目内的本地浏览器控制接口（BitBrowser），提供 createBrowser/openBrowser/closeBrowser。
# - 本模块当前不再依赖后端 src.auth 的 OAuth 回调流程，直接访问目标页面完成登录态校验。
# - log: 仓库根的日志封装模块，统一写入 RunLogs/ 并兼顾控制台输出。


# ==============================
# 仅本地测试用默认值（你测试 OK 后可整段注释/删除）
# ==============================
# TEST_DEFAULT_USERNAME = os.environ.get("RPA_TEST_USERNAME", "Nguyenminh13248@gmail.com")
# TEST_DEFAULT_PASSWORD = os.environ.get("RPA_TEST_PASSWORD", "thanhngan@22")

# BitBrowser：仅本地测试用默认窗口 ID。
# - 不传 --bitbrowser-id 时，将默认使用该 ID 连接现有窗口。
# - 若你想恢复默认行为（不指定 id，走 createBrowser()）：把下面这一行注释掉，或改为 ""/None。
TEST_DEFAULT_BITBROWSER_ID = os.environ.get(
    "RPA_TEST_BITBROWSER_ID", "f1ef6f8ccb6146988c67a63b92a78971"
)


# playwright-stealth 的 API 在不同版本中不一致：
# - 有的版本提供 `stealth_async(page)`
# - 有的版本只提供 `Stealth().apply_stealth_async(page)`
# 这里做兼容，避免因导入失败导致整个模块无法运行。
try:
    from playwright_stealth import stealth_async as _stealth_async  # type: ignore
except Exception:  # pragma: no cover
    _stealth_async = None

try:
    from playwright_stealth.stealth import Stealth as _Stealth  # type: ignore
except Exception:  # pragma: no cover
    _Stealth = None

_STEALTH = _Stealth() if _Stealth is not None else None


def _get_test_default_bitbrowser_id() -> Optional[str]:
    """获取测试默认 BitBrowser 窗口 ID。

    要求：只要 `TEST_DEFAULT_BITBROWSER_ID` 没被注释/置空，就优先生效。
    兼容：如果你把该常量整行注释/删除（导致变量不存在），这里也不会报错。
    """
    val = globals().get("TEST_DEFAULT_BITBROWSER_ID", None)
    if val is None:
        return None
    try:
        s = str(val).strip()
    except Exception:
        return None
    return s or None


# 说明：本函数用于在本地测试时优先复用一个固定的 BitBrowser 窗口 ID。12
# 在开发/调试时可以在文件顶部修改 `TEST_DEFAULT_BITBROWSER_ID` 或者通过环境变量
# RPA_TEST_BITBROWSER_ID 来指定。生产环境通常会注释掉该默认值以强制走 createBrowser().


async def _apply_stealth(page) -> None:
    if _stealth_async is not None:
        await _stealth_async(page)
        return
    if _STEALTH is not None:
        await _STEALTH.apply_stealth_async(page)
        return
    # 未安装或不可用时：静默降级
    return


# _apply_stealth 说明：
# - stealth 相关库用于注入防检测脚本（例如修改 navigator、WebGL 指纹等）以减少被网站
#   识别为自动化脚本的概率。该功能依赖第三方库 `playwright-stealth`，不同版本的 API
#   不一致，因此这里做了兼容处理（stealth_async 或 Stealth().apply_stealth_async）。
# - 如果未安装该库，函数会静默返回，不影响主流程。

# 兼容两种运行方式：
# 1) 在仓库根目录运行：python -m Rpa.BrowserAutomation.main
# 2) 在当前目录运行：cd Rpa/BrowserAutomation && python main.py
# 第二种情况下，顶层的 log.py 不在 sys.path，需要手动把仓库根目录加进去。
try:
    from log import log  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    try:
        from log import log  # type: ignore
    except ModuleNotFoundError:
        # 兼容当前项目没有 log.py 的情况：降级到标准 logging，避免因日志模块缺失中断主流程。
        import logging as _py_logging

        _fallback_logger = _py_logging.getLogger("Rpa.BrowserAutomation")

        class _FallbackLog:
            def info(self, message):
                _fallback_logger.info(str(message))

            def warning(self, message):
                _fallback_logger.warning(str(message))

            def error(self, message):
                _fallback_logger.error(str(message))

            def exception(self, message):
                _fallback_logger.exception(str(message))

        log = _FallbackLog()


def _print_and_log(message: str, *, level: str = "info") -> None:
    """同时输出到控制台与日志文件。

    说明：
    - print 方便 CLI/调试即时观察
    - log.* 便于后续排查/留痕
    """
    msg = str(message)
    try:
        print(msg)
    except Exception:
        pass
    try:
        fn = getattr(log, (level or "info").lower().strip(), None)
        if callable(fn):
            fn(msg)
        else:
            log.info(msg)
    except Exception:
        pass


def _print_and_log_exception(message: str) -> None:
    """在 except 分支同时打印并记录异常堆栈。"""
    msg = str(message)
    try:
        print(msg)
    except Exception:
        pass
    try:
        log.exception(msg)
    except Exception:
        try:
            log.error(msg)
        except Exception:
            pass


@dataclass(frozen=True)
class ValidateOptions:
    """验活参数。

    - headless: 是否无头（服务器/CI 推荐 True；本机调试推荐 False  ）
    - manual: 是否允许人工介入（遇到验证码/2FA 时很有用）
    - timeout_sec: 整体超时（等待进入目标页面）
    - slow_mo_ms: 让浏览器操作变慢，便于观察调试
    - user_agent: 自定义 UA（留空则使用浏览器默认 UA）
    - locale: 浏览器语言环境（如 zh-CN）
    - timezone_id: 时区（如 Asia/Shanghai）
    - viewport: 视口大小（留空则使用默认）
    - external_browser: 使用系统默认浏览器完成 OAuth（不经 Playwright 打开登录页）
    - human_delay_min_sec/human_delay_max_sec: 模拟“人工停顿”的随机延迟范围（秒）
    """

    headless: bool = False
    manual: bool = True
    timeout_sec: int = 300
    slow_mo_ms: int = 0
    user_agent: Optional[str] = (
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    )

    # locale: str = "zh-CN"
    # timezone_id: str = "Asia/Shanghai"

    locale: str = None
    timezone_id: str = None
    viewport_width: Optional[int] = 1366
    viewport_height: Optional[int] = 768
    external_browser: bool = False
    human_delay_min_sec: float = 4.0
    human_delay_max_sec: float = 6.0

    # BitBrowser（本地服务）：
    # - bitbrowser=True 时：自动 createBrowser() -> openBrowser() -> connect_over_cdp(ws)
    # - bitbrowser_id：指定窗口 id（传了就不会 createBrowser）
    # - bitbrowser_auto_delete=True 时：结束后 deleteBrowser(id)
    bitbrowser: bool = True
    bitbrowser_id: Optional[str] = None
    bitbrowser_auto_delete: bool = False


# ValidateOptions 说明（中文，便于理解）
# - headless: 是否无头运行。服务器/容器推荐 True，本地调试可设 False 以便观察操作。
# - manual: 是否允许人工介入（当遇到验证码/2FA 时可手动完成后续步骤）。
# - timeout_sec: 等待进入目标页面的整体超时时间，单位秒。
# - slow_mo_ms: Playwright 操作时每步放慢的毫秒数，便于调试。
# - user_agent/locale/timezone_id/viewport_*: 控制浏览器环境，用于降低因环境差异带来的风控概率。
# - external_browser: 若 True，使用系统默认浏览器打开授权链接（不使用 Playwright 打开），适合降低风控。
# - human_delay_min_sec/max_sec: 模拟用户在界面上的停顿，减少触发自动化检测的概率。
# - bitbrowser*: 与项目内 BitBrowser 服务交互的选项（本模块推荐使用 BitBrowser/CDP）。


class PlaywrightNotInstalled(RuntimeError):
    pass


# 控制面板“一键验活”默认配置（偏向稳定/可人工介入）。
# CLI/调用方仍可按需覆盖。
DEFAULT_PANEL_VALIDATE_OPTIONS = ValidateOptions(
    headless=False,
    manual=True,
    timeout_sec=300,
    slow_mo_ms=0,
    user_agent=None,
    locale="zh-CN",
    timezone_id="Asia/Shanghai",
    viewport_width=1366,
    viewport_height=768,
    external_browser=False,
    human_delay_min_sec=4.0,
    human_delay_max_sec=6.0,
)


def _normalize_locale_tz(options: ValidateOptions) -> tuple[str, str]:
    # Playwright 的 locale/timezone_id 不能为 None，这里给一个稳定默认值
    locale = (options.locale or "").strip() or "zh-CN"
    tz = (options.timezone_id or "").strip() or "Asia/Shanghai"
    return locale, tz


async def _human_delay(options: ValidateOptions, *, reason: str = "") -> None:
    min_s = float(options.human_delay_min_sec or 0.0)
    max_s = float(options.human_delay_max_sec or 0.0)
    if min_s <= 0 or max_s <= 0:
        return

    # 仅用于让流程更接近“人类操作节奏/页面加载节奏”，不保证绕过任何风控
    if reason:
        try:
            log.info(f"[RPA] human_delay: {reason}")
        except Exception:
            pass

    delay = random.uniform(min_s, max_s)
    await asyncio.sleep(delay)


# _human_delay 说明：
# - 用途：在动作之间插入随机停顿，模拟真实用户的节奏，降低被自动化检测或风控触发的概率。
# - 参数说明：options.human_delay_min_sec / human_delay_max_sec 控制最小/最大停顿秒数；
#   reason 用于日志，便于定位该停顿发生在流程的哪个阶段。


async def _human_type_text(
    page,
    selector: str,
    text: str,
    *,
    min_key_delay_ms: int = 60,
    max_key_delay_ms: int = 140,
    clear_first: bool = True,
    pre_delay_s: float = 0.4,
    post_delay_s: float = 0.6,
) -> None:
    """模拟“人类逐字输入”。

    说明：Playwright 的 fill() 会瞬间写入，容易显得非人类操作；这里改为逐字输入并插入轻微抖动。
    """
    selector = (selector or "").strip()
    if not selector:
        raise ValueError("selector 不能为空")

    loc = page.locator(selector).first
    await loc.wait_for(state="visible", timeout=10_000)
    await loc.click()
    # “人类停一下再打字”
    if pre_delay_s and pre_delay_s > 0:
        await asyncio.sleep(float(pre_delay_s))

    if clear_first:
        # 尽量清空旧内容（兼容不同平台快捷键）
        try:
            await loc.press("Control+A")
            await loc.press("Backspace")
        except Exception:
            try:
                await loc.press("Meta+A")
                await loc.press("Backspace")
            except Exception:
                try:
                    await loc.fill("")
                except Exception:
                    pass

    text = (text or "").strip()
    for ch in text:
        # 单字符输入（比一次性 type(text) 更“像人”）
        await page.keyboard.type(ch)
        key_delay = (
            random.uniform(float(min_key_delay_ms), float(max_key_delay_ms)) / 1000.0
        )
        await asyncio.sleep(key_delay)

    if post_delay_s and post_delay_s > 0:
        await asyncio.sleep(float(post_delay_s))


# _human_type_text 说明：
# - 目的：避免使用 Playwright 的一次性 fill/type 导致内容瞬间写入，显得像机器人；
#   所以逐字输入并在字符间插入随机延时，以提高“人类行为”相似度。
# - 参数要点：min_key_delay_ms/max_key_delay_ms 控制每个字符的随机延时；
#   clear_first 在输入前尽量清空旧内容（兼容不同平台的快捷键）。


def _build_context_kwargs(options: ValidateOptions) -> dict:
    locale, tz = _normalize_locale_tz(options)
    kwargs: dict = {
        "locale": locale,
        "timezone_id": tz,
        "accept_downloads": True,
    }
    if options.user_agent:
        kwargs["user_agent"] = options.user_agent
    if options.viewport_width and options.viewport_height:
        kwargs["viewport"] = {
            "width": int(options.viewport_width),
            "height": int(options.viewport_height),
        }
    return kwargs


def _extract_session_token_from_cookies(cookies: list[dict]) -> Optional[str]:
    """从 cookies 中提取 __Secure-next-auth.session-token（支持分片）。"""
    if not cookies:
        return None
    base_name = "__Secure-next-auth.session-token"

    # 1) 直接命中完整 cookie 名称
    for c in cookies:
        try:
            if str(c.get("name") or "") == base_name:
                v = str(c.get("value") or "").strip()
                if v:
                    return v
        except Exception:
            pass

    # 2) 分片命名兜底：__Secure-next-auth.session-token.0 / .1 / ...
    parts: list[tuple[int, str]] = []
    for c in cookies:
        try:
            name = str(c.get("name") or "")
            if not name.startswith(base_name + "."):
                continue
            suffix = name[len(base_name) + 1 :]
            idx = int(suffix)
            val = str(c.get("value") or "")
            parts.append((idx, val))
        except Exception:
            continue
    if parts:
        parts.sort(key=lambda x: x[0])
        joined = "".join(v for _, v in parts).strip()
        if joined:
            return joined
    return None


async def _collect_google_labs_cookies(context, *, entry_url: str) -> list[dict]:
    """尽力采集 labs/google 相关 cookies。"""
    urls = [
        (entry_url or "").strip() or "https://labs.google/fx/tools/flow",
        "https://labs.google/fx/tools/flow",
        "https://labs.google/fx/vi/tools/flow",
        "https://labs.google/",
        "https://www.google.com/",
    ]

    all_cookies: list[dict] = []
    # A) 按 URL 拉取（更接近浏览器扩展里的按页查询）
    for u in urls:
        if not u:
            continue
        try:
            cs = await context.cookies([u])
            if cs:
                all_cookies.extend(cs)
        except Exception:
            pass

    # B) 兜底：拿当前 context 全量 cookies
    try:
        cs_all = await context.cookies()
        if cs_all:
            all_cookies.extend(cs_all)
    except Exception:
        pass

    # 去重（name+domain+path）
    uniq: dict[str, dict] = {}
    for c in all_cookies:
        try:
            name = str(c.get("name") or "")
            domain = str(c.get("domain") or "")
            path = str(c.get("path") or "")
            key = f"{name}|{domain}|{path}"
            uniq[key] = c
        except Exception:
            continue
    return list(uniq.values())


def _cookie_header_for_domain(cookies: list[dict], domain_keyword: str) -> str:
    """将指定域名组 cookies 转成 Cookie Header 字符串（name=value; ...）。"""
    keyword = str(domain_keyword or "").strip().lower()
    if not keyword:
        return ""
    # 同名 cookie 以最后一个为准
    kv: dict[str, str] = {}
    for c in cookies or []:
        try:
            domain = str(c.get("domain") or "").strip().lower().lstrip(".")
            if not domain or not domain.endswith(keyword):
                continue
            name = str(c.get("name") or "").strip()
            value = str(c.get("value") or "")
            if name:
                kv[name] = value
        except Exception:
            continue
    if not kv:
        return ""
    return "; ".join(f"{k}={v}" for k, v in kv.items())


def _extract_email_from_cookies(cookies: list[dict], *, fallback: str = "") -> str:
    """尽力从 cookies 中提取邮箱，失败则回退到 fallback。"""
    for c in cookies or []:
        try:
            name = str(c.get("name") or "").strip().lower()
            if name not in ("email",):
                continue
            v = urllib_parse.unquote(str(c.get("value") or "").strip())
            if "@" in v:
                return v
        except Exception:
            continue
    fb = str(fallback or "").strip()
    return fb if "@" in fb else ""


def _try_extract_project_id_from_candidate(candidate_url: str) -> Optional[str]:
    """从单条 URL 中提取 projectId（支持 path 与 query 参数）。"""
    u = str(candidate_url or "").strip()
    if not u:
        return None

    # 1) 常见 path：.../project/{id}
    m = re.search(r"/project/([a-zA-Z0-9\-]+)", u, flags=re.IGNORECASE)
    if m:
        pid = (m.group(1) or "").strip()
        if pid:
            return pid

    # 2) 常见 query：?projectId=... 或 ?projectid=...
    try:
        parsed = urllib_parse.urlparse(u)
        qs = urllib_parse.parse_qs(parsed.query, keep_blank_values=False)
        for key in ("projectId", "projectid", "project_id"):
            vals = qs.get(key) or []
            for v in vals:
                vv = str(v or "").strip()
                if vv:
                    return vv
    except Exception:
        pass

    # 3) 兜底：对 URL 解码后再次正则提取 query 里的 projectId
    try:
        decoded = urllib_parse.unquote(u)
        m2 = re.search(r"(?:[?&])projectId=([a-zA-Z0-9\-]+)", decoded, flags=re.IGNORECASE)
        if m2:
            pid = (m2.group(1) or "").strip()
            if pid:
                return pid
    except Exception:
        pass
    return None


def _extract_project_ids_from_project_search_response_text(resp_text: str) -> list[str]:
    """从 project.searchUserProjects 响应文本中提取所有 projectId。"""
    text = str(resp_text or "")
    if not text:
        return []

    # JSON 结构可能存在变体，使用正则兜底提取 projectId 字段。
    found = re.findall(r'"projectId"\s*:\s*"([a-zA-Z0-9\-]+)"', text)
    uniq: list[str] = []
    for pid in found:
        p = str(pid or "").strip()
        if p and p not in uniq:
            uniq.append(p)
    return uniq


def _extract_project_id_from_url(
    url: str,
    *,
    project_ids_from_api: Optional[list[str]] = None,
    next_data_request_urls: Optional[list[str]] = None,
) -> Optional[str]:
    """提取 projectId（优先：project.searchUserProjects 响应 > 当前 URL > _next/data 请求）。"""
    u = str(url or "").strip()
    _print_and_log(f"[RPA][projectId] 开始提取，当前 URL: {u or '<empty>'}")

    # 最高优先级：project.searchUserProjects 响应中直接返回的 projectId 列表
    api_ids: list[str] = []
    for pid in (project_ids_from_api or []):
        p = str(pid or "").strip()
        if p and p not in api_ids:
            api_ids.append(p)
    if api_ids:
        chosen = api_ids[0]
        _print_and_log(f"[RPA][projectId] 从 project.searchUserProjects 响应提取成功: {chosen}")
        _print_and_log(f"[RPA][projectId] project.searchUserProjects 候选数: {len(api_ids)}")
        return chosen

    # 次优先：当前页面 URL
    pid = _try_extract_project_id_from_candidate(u)
    if pid:
        _print_and_log(f"[RPA][projectId] 从当前 URL 提取成功: {pid}")
        return pid
    _print_and_log("[RPA][projectId] 当前 URL 未提取到 projectId，尝试 _next/data 请求兜底", level="warning")

    # 兜底：_next/data 请求 URL（倒序优先最近请求）
    req_urls = list(next_data_request_urls or [])
    if not req_urls:
        _print_and_log("[RPA][projectId] 未捕获到可用的 _next/data 请求 URL", level="warning")
        return None

    _print_and_log(f"[RPA][projectId] 已捕获请求数: {len(req_urls)}，开始倒序匹配")
    for req_url in reversed(req_urls):
        pid = _try_extract_project_id_from_candidate(req_url)
        if pid:
            _print_and_log(f"[RPA][projectId] 从请求 URL 提取成功: {pid}")
            _print_and_log(f"[RPA][projectId] 命中请求 URL: {req_url}")
            return pid
    _print_and_log("[RPA][projectId] 所有候选 URL 均未提取到 projectId", level="warning")
    return None


def _build_token_payload(
    *,
    cookies: list[dict],
    username: str,
    current_url: str,
    project_ids_from_api: Optional[list[str]] = None,
    next_data_request_urls: Optional[list[str]] = None,
    state: str,
) -> dict:
    """构造导出 JSON（兼容前端 payload 结构）。"""
    domain_parts = "labs.google"
    domain_file = "google.com"
    cookie_map = {
        domain_parts: _cookie_header_for_domain(cookies, domain_parts),
        domain_file: _cookie_header_for_domain(cookies, domain_file),
    }
    email = _extract_email_from_cookies(cookies, fallback=username)
    project_id = _extract_project_id_from_url(
        current_url,
        project_ids_from_api=project_ids_from_api,
        next_data_request_urls=next_data_request_urls,
    )
    log.info(f"[RPA] project_id：{project_id}（综合来源提取）")
    return {
        "email": email,
        "cookie": cookie_map[domain_parts],
        "cookieFile": cookie_map[domain_file],
        "projectId": project_id,
        "points": 12500,
        "state": 2,
        "level": 5,
    }


def _save_token_payload_to_tmp(payload: dict) -> str:
    """保存 payload 到仓库 tmp/Token 目录，返回文件绝对路径。"""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "tmp" / "Token"
    out_dir.mkdir(parents=True, exist_ok=True)
    email = str((payload or {}).get("email") or "").strip() or "unknown"
    safe_email = re.sub(r"[^a-zA-Z0-9_.@-]+", "_", email).replace("@", "_at_")
    filename = f"{safe_email}_{int(time.time())}.json"
    out_path = out_dir / filename
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(out_path)


async def validate_antigravity_account(
    *,
    username: str,
    password: str,
    user_session: str,
    is_2fa_enabled: bool = False,
    twofa_password: Optional[str] = None,
    options: Optional[ValidateOptions] = None,
) -> dict:
    """对单个账号执行浏览器登录校验（目标页：labs.google/fx）。"""

    # 说明：此函数是本模块的核心入口，用来对单个账号执行浏览器登录并验证是否进入目标页面。
    # - 参数：
    #   - username/password: Google 登录凭据（用于自动化尝试登录）。
    #   - user_session: 会话标识（保留参数位，便于上层调用链追踪）。
    #   - options: ValidateOptions 对象，包含 headless/manual/timeout 等控制项。
    # - 返回：一个 dict，包含 success/message/error 等字段。
    # - 行为要点：
    #   1) 延迟导入 playwright，避免在未安装 playwright 的环境中提前 import 导致程序崩溃。
    #   2) 优先使用 BitBrowser + CDP 模式（通过本地服务拿到 ws，再用 Playwright connect_over_cdp 连接），
    #      因为直接用 Playwright 自行启动的浏览器在 Google 场景下更容易触发风控。
    #   3) 支持 external_browser 模式：直接用系统默认浏览器打开 URL，适合本机手工介入。
    #   4) 在关键 except 分支记录详细日志，便于在生产环境排查失败原因（例如 openBrowser 未返回 ws）。

    if not username or not str(username).strip():
        raise ValueError("username 不能为空")
    if password is None or not str(password).strip():
        raise ValueError("password 不能为空")
    if not user_session or not str(user_session).strip():
        raise ValueError("user_session 不能为空")

    options = options or ValidateOptions()

    # 延迟导入：避免在没有 playwright 的环境里导入就崩
    try:
        from playwright.async_api import (
            async_playwright,
            TimeoutError as PwTimeoutError,
        )
    except Exception as e:  # pragma: no cover
        raise PlaywrightNotInstalled(
            "未安装 Playwright。请先执行: pip install playwright && playwright install"
        ) from e

    entry_url = (
        os.getenv("ACCOUNTPOOL_RPA_TARGET_URL", "https://labs.google/fx/tools/flow")
        or "https://labs.google/fx/tools/flow"
    ).strip()
    if not re.match(r"^https?://", entry_url, re.IGNORECASE):
        entry_url = f"https://{entry_url.lstrip('/')}"
    log.info(f"[RPA] 入口页面: {entry_url}")

    # 外部浏览器模式：不使用 Playwright 打开 Google 登录页，避免触发“此浏览器或应用可能不安全”。
    if options.external_browser:
        if options.headless:
            raise RuntimeError(
                "external_browser 模式不支持 headless，请关闭 --headless"
            )
        log.info("[RPA] external_browser=True：将使用系统默认浏览器打开目标页面")
        log.info(f"[RPA] 请在弹出的浏览器窗口完成登录：{entry_url}")
        try:
            webbrowser.open(entry_url, new=1, autoraise=True)
        except Exception as e:
            raise RuntimeError(f"无法打开系统浏览器: {e}")

        return {
            "success": True,
            "message": "external_browser_opened",
            "error": None,
            "file_path": None,
            "auto_detected_project": None,
        }

    timeout_ms = max(1, int(options.timeout_sec)) * 1000
    target_re = re.compile(r"^https://labs\.google/fx(?:/|\?|$)", re.IGNORECASE)
    _print_and_log(f"将在 {options.timeout_sec} 秒内等待进入目标页: {entry_url}")

    async with async_playwright() as p:
        # Google 对 headless 更敏感；若想更稳定，建议 headful + 持久化 profile（user_data_dir）。
        context = None
        browser = None
        page = None
        bitbrowser_id = None
        final_url = ""
        token_payload_file_path: Optional[str] = None
        token_payload_cookie: Optional[str] = None
        token_payload_email: Optional[str] = None
        project_id_request_urls: list[str] = []
        next_data_request_cookies: list[str] = []
        project_ids_from_api: list[str] = []
        request_capture_tasks: list[asyncio.Task] = []
        response_capture_tasks: list[asyncio.Task] = []
        _request_capture_handler = None
        _response_capture_handler = None

        try:
            # BitBrowser 模式（唯一模式）：通过本地服务 create/open 拿到 ws，再用 CDP 接入。
            # - options.bitbrowser=True：会自动 createBrowser()
            # - options.bitbrowser_id ：指定现有窗口 id（不会 create）
            bitbrowser_id = getattr(options, "bitbrowser_id", None)
            if not bitbrowser_id:
                # 无论从 CLI 还是 web.py 调用：只要 TEST_DEFAULT_BITBROWSER_ID 没被注释/置空，就优先使用它
                test_default_id = _get_test_default_bitbrowser_id()
                if test_default_id:
                    bitbrowser_id = test_default_id
                    _print_and_log(
                        f"[RPA] 将使用 TEST_DEFAULT_BITBROWSER_ID 复用窗口，ID: {bitbrowser_id}"
                    )
            use_bitbrowser = bool(getattr(options, "bitbrowser", True) or bitbrowser_id)
            if not use_bitbrowser:
                raise RuntimeError(
                    "当前版本已移除 Playwright 自行启动浏览器；请使用 BitBrowser（options.bitbrowser=True 或指定 bitbrowser_id）"
                )
            if options.headless:
                raise RuntimeError("BitBrowser 模式不支持 headless，请关闭 --headless")
            try:
                from Rpa.Login.bit_api import createBrowser, openBrowser, closeBrowser
            except Exception:
                # 兼容在 Rpa/BrowserAutomation 目录直接运行时的导入路径
                from ..Login.bit_api import createBrowser, openBrowser, closeBrowser  # type: ignore

            if not bitbrowser_id:
                _print_and_log("[RPA] 将通过 BitBrowser 创建新的浏览器窗口...")
                bitbrowser_id = createBrowser()
                _print_and_log(f"[RPA] 已通过 BitBrowser 创建窗口，ID: {bitbrowser_id}")

            res = openBrowser(bitbrowser_id)
            _print_and_log(f"[RPA] 已通过 BitBrowser 打开窗口，ID: {bitbrowser_id}")
            ws = (res or {}).get("data", {}).get("ws")
            if not ws:
                raise RuntimeError(f"openBrowser 未返回 ws，响应: {res}")

            chromium = p.chromium
            browser = await chromium.connect_over_cdp(ws)
            if not browser.contexts:
                raise RuntimeError("BitBrowser/CDP 连接成功但没有可用的 context")
            context = browser.contexts[0]
            _print_and_log(
                f"[RPA] 已通过 CDP 连接到 BitBrowser 窗口，准备打开目标页面..."
            )

            def _remember_next_data_cookie(cookie_header: Optional[str]) -> None:
                text = str(cookie_header or "").strip()
                if not text:
                    return
                next_data_request_cookies.append(text)
                if len(next_data_request_cookies) > 300:
                    del next_data_request_cookies[0 : len(next_data_request_cookies) - 300]

            def _extract_cookie_from_headers_obj(headers_obj) -> Optional[str]:
                if headers_obj is None:
                    return None
                if callable(headers_obj):
                    try:
                        headers_obj = headers_obj()
                    except Exception:
                        return None
                if isinstance(headers_obj, dict):
                    for k, v in headers_obj.items():
                        if str(k or "").strip().lower() == "cookie":
                            text = str(v or "").strip()
                            return text or None
                return None

            async def _capture_next_data_request_cookie_async(request) -> None:
                cookie_header: Optional[str] = None
                try:
                    if hasattr(request, "all_headers"):
                        headers = await request.all_headers()
                        cookie_header = _extract_cookie_from_headers_obj(headers)
                except Exception:
                    pass
                if not cookie_header:
                    try:
                        if hasattr(request, "header_value"):
                            cookie_header = str(await request.header_value("cookie") or "").strip() or None
                    except Exception:
                        pass
                _remember_next_data_cookie(cookie_header)

            def _capture_request_for_project_id(request) -> None:
                try:
                    req_url = str(getattr(request, "url", "") or "").strip()
                    if not req_url:
                        return
                    req_url_l = req_url.lower()
                    if ("labs.google/fx/_next/data/" not in req_url_l) and ("projectid=" not in req_url_l):
                        return
                    project_id_request_urls.append(req_url)
                    if len(project_id_request_urls) > 300:
                        del project_id_request_urls[0 : len(project_id_request_urls) - 300]
                    if "labs.google/fx/_next/data/" in req_url_l:
                        # 优先尝试同步读取请求头中的 Cookie。
                        _remember_next_data_cookie(
                            _extract_cookie_from_headers_obj(getattr(request, "headers", None))
                        )
                        # 异步兜底：某些版本仅在 all_headers()/header_value 中可拿到完整头。
                        try:
                            task = asyncio.create_task(_capture_next_data_request_cookie_async(request))
                            request_capture_tasks.append(task)
                            if len(request_capture_tasks) > 600:
                                request_capture_tasks[:] = [x for x in request_capture_tasks if not x.done()]
                        except Exception:
                            pass
                except Exception:
                    pass

            async def _capture_project_search_response(response) -> None:
                try:
                    resp_url = str(getattr(response, "url", "") or "").strip()
                    if not resp_url:
                        return
                    resp_url_l = resp_url.lower()
                    if "labs.google/fx/api/trpc/project.searchuserprojects" not in resp_url_l:
                        return

                    text = await response.text()
                    ids = _extract_project_ids_from_project_search_response_text(text)
                    if not ids:
                        _print_and_log(
                            "[RPA][projectId] project.searchUserProjects 响应已捕获，但未解析到 projectId",
                            level="warning",
                        )
                        return

                    added = 0
                    for pid in ids:
                        p = str(pid or "").strip()
                        if p and p not in project_ids_from_api:
                            project_ids_from_api.append(p)
                            added += 1
                    if len(project_ids_from_api) > 300:
                        del project_ids_from_api[0 : len(project_ids_from_api) - 300]

                    _print_and_log(
                        f"[RPA][projectId] 从 project.searchUserProjects 响应提取 projectId {added} 个，累计 {len(project_ids_from_api)} 个"
                    )
                    _print_and_log(f"[RPA][projectId] project.searchUserProjects 命中 URL: {resp_url}")
                except Exception as e:
                    _print_and_log_exception(
                        f"[RPA][projectId] 解析 project.searchUserProjects 响应失败: {type(e).__name__}: {e}"
                    )

            def _capture_response_for_project_id(response) -> None:
                try:
                    t = asyncio.create_task(_capture_project_search_response(response))
                    response_capture_tasks.append(t)
                    if len(response_capture_tasks) > 600:
                        # 清理已结束任务，避免列表无限增长
                        response_capture_tasks[:] = [x for x in response_capture_tasks if not x.done()]
                except Exception:
                    pass

            _request_capture_handler = _capture_request_for_project_id
            _response_capture_handler = _capture_response_for_project_id
            try:
                context.on("request", _request_capture_handler)
                context.on("response", _response_capture_handler)
                _print_and_log("[RPA][projectId] 已启用请求监听（_next/data + projectId）")
                _print_and_log("[RPA][projectId] 已启用响应监听（project.searchUserProjects）")
            except Exception as e:
                _print_and_log_exception(
                    f"[RPA][projectId] 启用请求监听失败: {type(e).__name__}: {e}"
                )
            page = await context.new_page()

            # await _apply_stealth(page)
            page = await _open_flow_and_click_create(
                page=page,
                context=context,
                entry_url=entry_url,
                options=options,
            )

            # 1) 尝试自动登录
            _print_and_log("[RPA] 步骤1/2: 尝试自动登录...")
            try:
                await _best_effort_google_login(
                    page,
                    username=username,
                    password=password,
                    options=options,
                    is_2fa_enabled=bool(is_2fa_enabled),
                    twofa_password=(str(twofa_password or "").strip() or None),
                )
                _print_and_log("[RPA] 步骤1/2: 自动登录完成")
            except Exception as e:
                _print_and_log_exception(
                    f"[RPA] 步骤1/2: 自动登录发生异常: {type(e).__name__}: {e}"
                )
                raise

            # 2) 等待进入目标页面（labs.google/fx）
            _print_and_log(
                f"[RPA] 步骤2/2: 开始等待进入目标页面，超时 {options.timeout_sec} 秒"
            )

            try:
                await page.wait_for_url(target_re, timeout=timeout_ms)
                final_url = page.url or ""
                _print_and_log(f"[RPA] 步骤2/2: 已进入目标页面 URL: {final_url}")
            except PwTimeoutError:
                if options.manual:
                    _print_and_log(
                        "[RPA] 步骤2/2: 等待目标页面超时，进入人工介入等待...",
                        level="warning",
                    )
                    await _wait_until_target_url(page, target_re, timeout_ms=timeout_ms)
                    final_url = page.url or ""
                    _print_and_log(
                        f"[RPA] 步骤2/2: 人工介入后已进入目标页面 URL: {final_url}"
                    )
                else:
                    _print_and_log(
                        "[RPA] 步骤2/2: 等待目标页面超时且未允许人工介入，抛出超时错误",
                        level="error",
                    )
                    raise

            # 等待页面渲染出成功字样（非必须，但可减少“URL 刚跳过去内容还没刷新”的误判）
            try:
                await page.wait_for_timeout(300)
            except Exception:
                pass

            # 进入目标页面后，随机等待一小段时间再继续，避免过快关闭窗口
            try:
                delay_s = random.uniform(2.0, 6.0)
                _print_and_log(f"[RPA] 进入目标页后停留 {delay_s:.1f}s 再继续...")
                await asyncio.sleep(delay_s)
            except Exception:
                pass

            # 3) 从当前浏览器上下文提取 __Secure-next-auth.session-token
            _print_and_log("[RPA] 开始提取 __Secure-next-auth.session-token ...")
            session_token: Optional[str] = None
            try:
                # 等待已触发的 project.searchUserProjects 响应解析任务完成，提升 projectId 命中率
                try:
                    pending_req_tasks = [t for t in request_capture_tasks if not t.done()]
                    if pending_req_tasks:
                        await asyncio.gather(*pending_req_tasks, return_exceptions=True)
                except Exception:
                    pass
                try:
                    pending_tasks = [t for t in response_capture_tasks if not t.done()]
                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                except Exception:
                    pass

                all_cookies = await _collect_google_labs_cookies(
                    context, entry_url=entry_url
                )
                try:
                    cookie_brief = [
                        {
                            "name": str(c.get("name") or ""),
                            "domain": str(c.get("domain") or ""),
                        }
                        for c in all_cookies
                    ]
                    _print_and_log(f"[RPA] Cookies 总数: {len(all_cookies)}")
                    _print_and_log(f"[RPA] Cookie 名称预览: {cookie_brief[:20]}")
                except Exception:
                    pass
                session_token = _extract_session_token_from_cookies(all_cookies)
                if session_token:
                    _print_and_log(
                        f"[RPA] 已提取 __Secure-next-auth.session-token，长度: {len(session_token)}"
                    )
                    _print_and_log(
                        f"[RPA] __Secure-next-auth.session-token 内容: {session_token}"
                    )
                else:
                    _print_and_log(
                        "[RPA] 未提取到 __Secure-next-auth.session-token，请确认已完成登录并进入目标页面",
                        level="warning",
                    )

                # 4) 导出完整 token（labs.google + google.com）到 tmp/Token/*.json
                try:
                    payload = _build_token_payload(
                        cookies=all_cookies,
                        username=username,
                        current_url=(page.url or final_url or entry_url),
                        project_ids_from_api=project_ids_from_api,
                        next_data_request_urls=project_id_request_urls,
                        state=2,
                    )
                    cookie_from_next_data = (
                        str(next_data_request_cookies[-1] or "").strip()
                        if next_data_request_cookies
                        else ""
                    )
                    cookie_from_payload = str((payload or {}).get("cookie") or "").strip()
                    token_payload_cookie = cookie_from_next_data or cookie_from_payload or None
                    if cookie_from_next_data:
                        _print_and_log(
                            f"[RPA] token_payload_cookie 已切换为 _next/data 请求头 Cookie，长度: {len(cookie_from_next_data)}"
                        )
                    elif cookie_from_payload:
                        _print_and_log(
                            f"[RPA] 未捕获到 _next/data 请求头 Cookie，回退到导出 payload.cookie，长度: {len(cookie_from_payload)}",
                            level="warning",
                        )
                    else:
                        _print_and_log(
                            "[RPA] 未获取到可用 cookie（_next/data 请求头与 payload.cookie 都为空）",
                            level="warning",
                        )
                    token_payload_email = (
                        str((payload or {}).get("email") or "").strip() or None
                    )
                    token_payload_file_path = _save_token_payload_to_tmp(payload)
                    _print_and_log(
                        f"[RPA] 已保存完整 Token JSON: {token_payload_file_path}"
                    )
                except Exception as e:
                    _print_and_log_exception(
                        f"[RPA] 保存完整 Token JSON 失败: {type(e).__name__}: {e}"
                    )
            except Exception as e:
                _print_and_log_exception(
                    f"[RPA] 提取 session-token 失败: {type(e).__name__}: {e}"
                )
                session_token = None

        finally:
            # BitBrowser 模式下：page/context 是挂在外部浏览器实例上的，通常不建议 close context。
            # 这里保持与原逻辑一致：优先关闭 playwright 侧对象；BitBrowser 窗口用 closeBrowser 关闭。
            try:
                if page is not None:
                    log.info("[RPA] 关闭 Playwright page")
                    await page.close()
            except Exception:
                try:
                    log.exception("[RPA] 关闭 Playwright page 失败")
                except Exception:
                    pass
            try:
                if context is not None:
                    log.info("[RPA] 关闭 Playwright context")
                    await context.close()
            except Exception:
                try:
                    log.exception("[RPA] 关闭 Playwright context 失败")
                except Exception:
                    pass
            try:
                if browser is not None:
                    log.info("[RPA] 关闭 Playwright browser")
                    await browser.close()
            except Exception:
                try:
                    log.exception("[RPA] 关闭 Playwright browser 失败")
                except Exception:
                    pass
            if bitbrowser_id:
                try:
                    try:
                        from Rpa.Login.bit_api import closeBrowser, deleteBrowser
                    except Exception:
                        from ..Login.bit_api import closeBrowser, deleteBrowser  # type: ignore
                    closeBrowser(bitbrowser_id)
                    _print_and_log(
                        f"[RPA] 已通过 BitBrowser 关闭窗口，ID: {bitbrowser_id}"
                    )
                    # 可选：自动删除窗口
                    if bool(getattr(options, "bitbrowser_auto_delete", False)):
                        deleteBrowser(bitbrowser_id)
                        _print_and_log(
                            f"[RPA] 已通过 BitBrowser 删除窗口，ID: {bitbrowser_id}"
                        )
                except Exception:
                    try:
                        log.exception(
                            f"[RPA] BitBrowser 关闭/删除窗口失败，ID: {bitbrowser_id}"
                        )
                    except Exception:
                        pass
            try:
                if context is not None and _request_capture_handler is not None:
                    context.remove_listener("request", _request_capture_handler)
            except Exception:
                pass
            try:
                if context is not None and _response_capture_handler is not None:
                    context.remove_listener("response", _response_capture_handler)
            except Exception:
                pass

    # 4) 返回登录校验结果
    return {
        "success": True,
        "message": (
            "rpa_login_success"
            if session_token
            else "rpa_login_success_but_no_session_token"
        ),
        "error": None,
        "file_path": token_payload_file_path,
        "auto_detected_project": True,
        "session_token": (session_token or None),
        "cookie": token_payload_cookie,
        "payload_email": token_payload_email,
    }


async def _wait_until_target_url(
    page, target_re: re.Pattern, *, timeout_ms: int
) -> None:
    """人工介入时的等待：每 500ms 检查一次是否进入目标 URL。"""
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        try:
            if target_re.match(page.url or ""):
                return
        except Exception:
            pass
        await page.wait_for_timeout(500)
    raise TimeoutError("等待进入目标页面超时（可能卡在验证码/2FA/授权页）")


async def _open_flow_and_click_create(
    *,
    page,
    context,
    entry_url: str,
    options,
):
    """打开 flow 页面并点击 Create with Flow，随后返回应继续操作的 page。"""
    await page.goto(entry_url, wait_until="domcontentloaded")
    await _wait_for_flow_entry_ready(page, timeout_ms=18000)
    await _human_delay(options, reason="after_open_flow_entry")
    _print_and_log(f"--------打开入口页面: {entry_url}")

    cur_url = (page.url or "").lower()
    if "accounts.google.com" in cur_url:
        _print_and_log("[RPA] 已进入 Google 登录页，跳过 Create with Flow 点击")
        return page

    selectors = [
        "button:has-text('Create with Flow')",
        "button:has(span:has-text('Create with Flow'))",
        "button.sc-16c4830a-1.sc-c0d0216b-0",
    ]
    clicked = await _click_first_visible(page, selectors, timeout_ms=12000)

    if not clicked:
        try:
            clicked = bool(
                await page.evaluate(
                    """() => {
						const spans = Array.from(document.querySelectorAll("span"));
						const target = spans.find(s => ((s.textContent || "").trim() === "Create with Flow"));
						if (!target) return false;
						const btn = target.closest("button");
						if (!btn) return false;
						btn.click();
						return true;
					}"""
                )
            )
        except Exception:
            clicked = False

    if not clicked:
        _print_and_log(
            "[RPA] 未找到 Create with Flow 按钮，继续后续登录流程", level="warning"
        )
        return page

    _print_and_log("[RPA] 已点击 Create with Flow，等待跳转到登录/授权页面...")
    try:
        await page.wait_for_timeout(1200)
    except Exception:
        pass
    await _human_delay(options, reason="after_click_create_with_flow")

    try:
        pages = list(getattr(context, "pages", []) or [])
        if pages and pages[-1] is not page:
            page = pages[-1]
            try:
                await page.bring_to_front()
            except Exception:
                pass
            _print_and_log(f"[RPA] 检测到新标签页，已切换 URL: {page.url}")
    except Exception:
        pass

    return page


async def _wait_for_flow_entry_ready(page, *, timeout_ms: int = 15000) -> None:
    """等待 flow 入口页基本加载完成，减少重页面导致的点击失败。"""
    try:
        await page.wait_for_load_state(
            "domcontentloaded", timeout=min(timeout_ms, 10000)
        )
    except Exception:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 8000))
    except Exception:
        # 某些页面有长连接，networkidle 不一定能稳定达成；这里静默降级到轮询检测。
        pass

    deadline = time.time() + (max(1000, int(timeout_ms)) / 1000.0)
    selectors = [
        "button:has-text('Create with Flow')",
        "button:has(span:has-text('Create with Flow'))",
        "button.sc-16c4830a-1.sc-c0d0216b-0",
    ]
    while time.time() < deadline:
        try:
            cur_url = (page.url or "").lower()
            if "accounts.google.com" in cur_url:
                return
        except Exception:
            pass

        try:
            ready = await page.evaluate("() => document.readyState")
            if str(ready).lower() not in ("interactive", "complete"):
                await page.wait_for_timeout(250)
                continue
        except Exception:
            pass

        for sel in selectors:
            try:
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return
            except Exception:
                pass
        try:
            await page.wait_for_timeout(250)
        except Exception:
            pass

    _print_and_log(
        "[RPA] Flow 入口页加载较慢，未确认按钮可见，继续尝试点击", level="warning"
    )


# _wait_until_target_url 说明：
# - 用于在 manual 模式下，让人工在浏览器中完成流程后由脚本轮询检测 URL 是否进入目标地址；
# - 采用短轮询（500ms）以便及时发现进入目标页面，但总超时仍受外层 timeout_ms 控制。


class AccountInvalidError(RuntimeError):
    """账号在登录阶段被判定为不可用（验证码/手机号验证/风控挑战等）。"""


async def _detect_google_challenge_reason(page) -> Optional[str]:
    """快速检测 Google 登录/授权过程中是否出现挑战页。

    命中则返回原因字符串；否则返回 None。

    说明：尽量使用 DOM marker + 关键词扫描，避免 wait_for 导致变慢。
    """
    # 1) DOM markers（最快）
    marker_selectors = [
        "iframe[title*='recaptcha' i]",
        "iframe[src*='recaptcha' i]",
        "textarea#g-recaptcha-response",
        "div.g-recaptcha",
        "div[data-site-key]",
        "#captcha",
    ]
    for sel in marker_selectors:
        try:
            if await page.locator(sel).count() > 0:
                return "检测到验证码/挑战组件"
        except Exception:
            pass

    # 2) 文案 markers（多语言兜底）
    text_markers = [
        "证明您不是自动程序",
        "请验证您并非自动程序",
        "尝试的失败次数过多",
        "验证您并非自动程序",
        "我不是机器人",
        "验证您的电话号码",
        "验证是您本人",
        "确认是您本人",
        "Verify it's you",
        "Verify it’s you",
        "Verify you",
        "phone number",
        "Verify your phone",
        "I'm not a robot",
        "I am not a robot",
        "reCAPTCHA",
        "captcha",
        "Too many failed attempts",
        "Try again later",
        "unusual activity",
        "suspicious",
    ]
    try:
        # inner_text 比 page.content() 更轻量，且便于关键词匹配
        body_text = await page.locator("body").inner_text(timeout=5000)
        body_norm = (body_text or "").lower()
        for t in text_markers:
            if (t or "").strip() and (t.lower() in body_norm):
                # 返回命中的关键词作为原因
                return f"页面提示：{t}"
    except Exception:
        pass

    return None


async def _raise_if_google_challenge(
    page, *, stage: str = "", allow_2fa: bool = False
) -> None:
    # 当前项目默认不启用“挑战页即失败”判定，避免误杀正常流程。
    # 如需恢复旧行为，可设置环境变量 ACCOUNTPOOL_RPA_ENABLE_CHALLENGE_BLOCK=1。
    enable_block = str(
        os.getenv("ACCOUNTPOOL_RPA_ENABLE_CHALLENGE_BLOCK", "0")
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enable_block:
        return

    if allow_2fa and await _is_google_2fa_page(page):
        return
    reason = await _detect_google_challenge_reason(page)
    if not reason:
        return
    stage_msg = f"（阶段: {stage}）" if stage else ""
    raise AccountInvalidError(
        f"出现了意外的验证码|手机号验证，账号可能已作废：{reason}{stage_msg}"
    )


async def _best_effort_google_login(
    page,
    *,
    username: str,
    password: str,
    options: ValidateOptions,
    is_2fa_enabled: bool = False,
    twofa_password: Optional[str] = None,
) -> None:
    """尽力自动完成 Google 登录。

    注意：这里不追求覆盖所有分支（2FA/验证码/设备验证等），只做常见路径的 best-effort。
    """

    # 若一上来就是挑战页/风控页，直接判定账号不可用（避免在 best-effort 的 try/except 里被吞掉）
    # await _raise_if_google_challenge(page, stage="login_start")

    # Google 的登录页面多数情况下会出现 #identifierId + #identifierNext
    email_page_present = False
    try:
        email_sel = await _wait_for_google_identifier_step(page, timeout_ms=16000)
        if email_sel:
            email_page_present = True
            ok = await _set_input_value_robust(
                page,
                selectors=[
                    email_sel,
                    "input#identifierId",
                    "input[name='identifier']",
                    "input[type='email'][aria-label='Email or phone']",
                    "div.Xb9hP input[name='identifier']",
                ],
                text=username,
                timeout_ms=12000,
            )
            if not ok:
                raise RuntimeError("邮箱输入失败：identifier 输入框未成功写入")

            await _raise_if_google_challenge(page, stage="after_fill_email")
            await _human_delay(options, reason="after_fill_email")
            clicked = await _click_first_visible(
                page,
                [
                    "#identifierNext",
                    "button:has-text('下一步')",
                    "button:has-text('Next')",
                    "div[role='button']:has-text('Next')",
                    "button[jsname='LgbsSe']:has-text('Next')",
                ],
                timeout_ms=6000,
            )
            if not clicked:
                try:
                    await page.locator(email_sel).first.press("Enter")
                except Exception:
                    pass

            await _raise_if_google_challenge(page, stage="after_click_email_next")
            arrived = await _wait_for_google_password_step(page, timeout_ms=15000)
            if not arrived:
                # 再补一次触发，避免首击丢失
                try:
                    await page.locator("#identifierNext").first.click(timeout=1500)
                except Exception:
                    pass
                try:
                    await page.locator(email_sel).first.press("Enter")
                except Exception:
                    pass
                arrived = await _wait_for_google_password_step(page, timeout_ms=8000)
            if not arrived:
                raise RuntimeError("点击邮箱下一步后未进入密码页")

            await _human_delay(options, reason="after_click_email_next")
    except AccountInvalidError:
        raise
    except Exception:
        # 没有邮箱页时忽略；如果确实在邮箱页则抛出，避免“卡住但流程继续”。
        if email_page_present:
            raise

    # 密码页
    password_page_present = False
    try:
        pwd_sel = "input[type='password'], input[name='Passwd']"
        if await page.locator(pwd_sel).count() > 0:
            password_page_present = True
            ok = await _set_input_value_robust(
                page,
                selectors=[
                    "input[type='password']",
                    "input[name='Passwd']",
                    pwd_sel,
                ],
                text=password,
                timeout_ms=12000,
            )
            if not ok:
                raise RuntimeError("密码输入失败：password 输入框未成功写入")

            await _raise_if_google_challenge(page, stage="after_fill_password")
            await _human_delay(options, reason="after_fill_password")
            clicked = await _click_first_visible(
                page,
                [
                    "#passwordNext",
                    "button:has-text('下一步')",
                    "button:has-text('Next')",
                    "div[role='button']:has-text('Next')",
                ],
                timeout_ms=6000,
            )
            if not clicked:
                try:
                    await page.locator(pwd_sel).first.press("Enter")
                except Exception:
                    pass
            await _raise_if_google_challenge(
                page, stage="after_click_password_next", allow_2fa=True
            )
            if is_2fa_enabled:
                detected_2fa = await _wait_for_google_2fa_page(page, timeout_ms=18_000)
                if detected_2fa:
                    _print_and_log("[RPA] 点击密码下一步后检测到 2FA 页面，开始自动填码")
                else:
                    _print_and_log(
                        f"[RPA] 点击密码下一步后未检测到 2FA 页面（当前 URL: {page.url}），继续后续流程",
                        level="warning",
                    )
                await _try_google_2fa_flow(
                    page,
                    twofa_password=(str(twofa_password or "").strip() or ""),
                    options=options,
                )
            await _human_delay(options, reason="after_click_password_next")
    except AccountInvalidError:
        raise
    except Exception:
        if password_page_present:
            raise

    # 有些场景会在密码提交后延迟跳到 2FA 页；这里再兜底尝试一次，避免提前结束。
    if is_2fa_enabled:
        try:
            detected_2fa = await _wait_for_google_2fa_page(page, timeout_ms=10_000)
            if detected_2fa:
                _print_and_log("[RPA] 登录尾阶段检测到 2FA 页面，执行兜底自动填码")
                await _try_google_2fa_flow(
                    page,
                    twofa_password=(str(twofa_password or "").strip() or ""),
                    options=options,
                )
        except Exception:
            # 尾阶段兜底不阻断主流程，但保留日志方便排查。
            _print_and_log_exception("[RPA] 登录尾阶段 2FA 兜底处理失败")

    # 登录流程结束前再检查一次（有些风控会在点击 Next 后延迟出现）
    await _raise_if_google_challenge(
        page, stage="login_end", allow_2fa=bool(is_2fa_enabled)
    )


async def _wait_for_google_password_step(page, *, timeout_ms: int = 12000) -> bool:
    """等待从邮箱页切到密码页。"""
    deadline = time.time() + (max(1000, int(timeout_ms)) / 1000.0)
    while time.time() < deadline:
        try:
            if (
                await page.locator(
                    "input[type='password'], input[name='Passwd']"
                ).count()
                > 0
            ):
                return True
        except Exception:
            pass
        try:
            url = (page.url or "").lower()
            if "challenge/pwd" in url:
                return True
        except Exception:
            pass
        try:
            await page.wait_for_timeout(250)
        except Exception:
            pass
    return False


async def _is_google_2fa_page(page) -> bool:
    """判断是否处于 Google 2FA 挑战流程页面。"""
    try:
        url = (page.url or "").lower()
        if any(
            x in url
            for x in [
                "/challenge/selection",
                "/challenge/totp",
                "/challenge/ipp",
                "/challenge/ootp",
            ]
        ):
            return True
    except Exception:
        pass
    try:
        body = (await page.locator("body").inner_text(timeout=2000) or "").lower()
        if any(
            x in body
            for x in [
                "2-step verification",
                "2-step",
                "google authenticator",
                "verification code",
                "security code",
                "try another way",
                "两步验证",
                "验证代码",
                "验证码",
            ]
        ):
            return True
    except Exception:
        pass
    return False


async def _wait_for_google_2fa_page(page, *, timeout_ms: int = 12_000) -> bool:
    """等待 Google 2FA 页面出现（用于密码 Next 后的过渡期）。"""
    deadline = time.time() + (max(200, int(timeout_ms)) / 1000.0)
    while time.time() < deadline:
        try:
            if await _is_google_2fa_page(page):
                return True
        except Exception:
            pass
        try:
            await page.wait_for_timeout(220)
        except Exception:
            pass
    return False


async def _wait_for_2fa_code_input_selector(
    page, *, timeout_ms: int = 12000
) -> Optional[str]:
    """等待 2FA 验证码输入框出现，返回可用 selector。"""
    selectors = [
        "input#totpPin",
        "input[name='totpPin']",
        "input[name='idvPin']",
        "input[autocomplete='one-time-code']",
        "input[inputmode='numeric']",
        "input[type='tel']",
        "input[aria-label*='code' i]",
        "input[aria-label*='验证码']",
    ]
    return await _find_2fa_code_input_selector(
        page, selectors=selectors, timeout_ms=timeout_ms
    )


async def _find_2fa_code_input_selector(
    page, *, selectors: list[str], timeout_ms: int = 12000
) -> Optional[str]:
    """按给定 selectors 检测 2FA 验证码输入框（支持快速/常规两种超时）。"""
    deadline = time.time() + (max(50, int(timeout_ms)) / 1000.0)
    vis_wait_ms = 180 if int(timeout_ms) <= 500 else 1000
    while time.time() < deadline:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await page.locator(sel).count() <= 0:
                    continue
                await loc.wait_for(state="visible", timeout=vis_wait_ms)
                return sel
            except Exception:
                pass
        try:
            await page.wait_for_timeout(80 if int(timeout_ms) <= 500 else 250)
        except Exception:
            pass
    return None


async def _quick_try_click_try_another_way(page) -> bool:
    """快速尝试点击 Try another way，避免 2FA 流程前置阻塞。"""
    selectors = [
        "button:has-text('Try another way')",
        "div[role='button']:has-text('Try another way')",
        "div[role='link']:has-text('Try another way')",
        "button:has-text('尝试其他方式')",
        "div[role='button']:has-text('尝试其他方式')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await page.locator(sel).count() <= 0:
                continue
            try:
                visible = await loc.is_visible(timeout=120)
            except Exception:
                visible = False
            if not visible:
                continue
            try:
                await loc.click(timeout=350)
            except Exception:
                await loc.click(timeout=350, force=True)
            return True
        except Exception:
            continue
    return False


def _fetch_2fa_code_from_cn(
    twofa_password: str, *, timeout_sec: int = 15
) -> Optional[str]:
    """调用 2fa.cn 接口获取动态验证码。"""
    secret = str(twofa_password or "").strip() or ""
    if not secret:
        return None
    url = f"https://2fa.cn/codes/{urllib_parse.quote(secret, safe='')}"
    req = urllib_request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="GET",
    )
    with urllib_request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw or "{}")
    code = str(data.get("code") or "").strip()
    if not code:
        return None
    if not re.fullmatch(r"\d{4,10}", code):
        return None
    return code


async def _try_google_2fa_flow(
    page, *, twofa_password: str, options: ValidateOptions
) -> None:
    """若出现 2FA 页面，自动调用 2fa.cn 拉取 code 并填入继续。"""
    secret = str(twofa_password or "").strip() or ""
    if not secret:
        _print_and_log(
            "[RPA] 2FA 已启用但未提供 twofa_password，跳过自动填码", level="warning"
        )
        return
    if not await _is_google_2fa_page(page):
        return

    stage_started = time.perf_counter()
    _print_and_log("[RPA] 检测到 Google 2FA 页面，尝试自动获取并填写验证码...")

    # 先尝试切换到“Google Authenticator app / verification code”路径
    try:
        # 快速点击：避免不存在该按钮时白等 1.8 秒
        clicked_try_another = await _quick_try_click_try_another_way(page)
        if clicked_try_another:
            try:
                await page.wait_for_timeout(220)
            except Exception:
                pass
    except Exception:
        pass

    before_select_ms = int((time.perf_counter() - stage_started) * 1000)
    _print_and_log(f"[RPA][2FA] 进入验证方式选择前耗时: {before_select_ms}ms")

    select_started = time.perf_counter()
    if not await _try_select_google_authenticator_option(page, timeout_ms=8500):
        _print_and_log(
            "[RPA] 未能可靠点击 Google Authenticator 选项，继续尝试检测验证码输入框",
            level="warning",
        )
    select_ms = int((time.perf_counter() - select_started) * 1000)
    _print_and_log(f"[RPA][2FA] 验证方式选择耗时: {select_ms}ms")

    code_input_sel = await _wait_for_2fa_code_input_selector(page, timeout_ms=12000)
    if not code_input_sel:
        _print_and_log("[RPA] 未检测到 2FA 验证码输入框，跳过自动填码", level="warning")
        return

    last_error: Optional[str] = None
    for attempt in range(2):
        _print_and_log(f"[RPA] 2FA 自动填码尝试 {attempt + 1}/2")
        try:
            code = await asyncio.to_thread(
                _fetch_2fa_code_from_cn, secret, timeout_sec=15
            )
        except Exception as e:
            code = None
            last_error = str(e)

        if not code:
            last_error = last_error or "2fa.cn 未返回有效 code"
            _print_and_log(f"[RPA] 2FA 获取 code 失败：{last_error}", level="warning")
            await asyncio.sleep(1.0)
            continue

        ok = await _set_input_value_robust(
            page,
            selectors=[
                code_input_sel,
                "input#totpPin",
                "input[name='totpPin']",
                "input[name='idvPin']",
                "input[autocomplete='one-time-code']",
                "input[inputmode='numeric']",
                "input[type='tel']",
            ],
            text=code,
            timeout_ms=5000,
        )
        if not ok:
            last_error = "验证码写入失败"
            await asyncio.sleep(0.8)
            continue

        clicked = await _click_first_visible(
            page,
            [
                "#totpNext",
                "#idvPreregisteredPhoneNext",
                "button:has-text('Next')",
                "div[role='button']:has-text('Next')",
                "button:has-text('下一步')",
            ],
            timeout_ms=3000,
        )
        if not clicked:
            try:
                await page.locator(code_input_sel).first.press("Enter")
            except Exception:
                pass

        try:
            await page.wait_for_timeout(1800)
        except Exception:
            pass

        # 若验证码输入框已消失，认为提交成功进入下一步
        next_code_input = await _wait_for_2fa_code_input_selector(page, timeout_ms=1200)
        if not next_code_input:
            _print_and_log("[RPA] 2FA 验证码已提交")
            return
        last_error = "验证码可能已过期或校验失败"

    raise RuntimeError(f"2FA 自动验证失败: {last_error or 'unknown error'}")


async def _try_select_google_authenticator_option(
    page, *, timeout_ms: int = 12000
) -> bool:
    """在 2FA 方式选择页点击 Google Authenticator / verification code 选项。"""
    # 如果已经在验证码输入页，直接返回
    quick_code_selectors = [
        "input#totpPin",
        "input[name='totpPin']",
        "input[name='idvPin']",
        "input[autocomplete='one-time-code']",
        "input[inputmode='numeric']",
        "input[type='tel']",
    ]
    if await _find_2fa_code_input_selector(
        page, selectors=quick_code_selectors, timeout_ms=180
    ):
        return True

    deadline = time.time() + (max(1000, int(timeout_ms)) / 1000.0)
    selectors = [
        # 最稳妥：基于 challenge 属性（动态 class 变化也不影响）
        "[data-action='selectchallenge'][data-challengetype='6']",
        "[jsname='EBHGs'][data-challengetype='6']",
        # 你提供的真实 DOM 结构（最优先）
        "div[role='link'][data-action='selectchallenge'][data-challengetype='6']",
        "div[jsname='EBHGs'][data-challengetype='6']",
        "div.VV3oRb[data-challengetype='6']",
        # 文案与结构兜底
        "div[role='link']:has(.l5PPKe strong:has-text('Google Authenticator'))",
        "div[role='link']:has(.l5PPKe:has-text('Google Authenticator'))",
        "div[role='link']:has-text('Google Authenticator')",
        "div[role='link']:has-text('verification code')",
        "div[role='link']:has-text('security code')",
        "div[role='button']:has-text('Google Authenticator')",
        "div[role='button']:has-text('verification code')",
    ]

    _print_and_log("[RPA][2FA] 尝试选择 Google Authenticator 验证方式...")

    # 先进行一次极快的 JS 直接点击（命中 data-challengetype=6 就直接触发）
    try:
        fast_clicked = bool(
            await page.evaluate(
                """() => {
					const target = document.querySelector("[data-action='selectchallenge'][data-challengetype='6'], [jsname='EBHGs'][data-challengetype='6']");
					if (!target) return false;
					try { target.scrollIntoView({block:'center'}); } catch (e) {}
					try { target.focus && target.focus(); } catch (e) {}
					target.dispatchEvent(new MouseEvent('mousedown', {bubbles:true}));
					target.dispatchEvent(new MouseEvent('mouseup', {bubbles:true}));
					target.click();
					target.dispatchEvent(new MouseEvent('click', {bubbles:true}));
					return true;
				}"""
            )
        )
        if fast_clicked and await _wait_for_2fa_code_input_selector(
            page, timeout_ms=2200
        ):
            return True
    except Exception:
        pass

    while time.time() < deadline:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await page.locator(sel).count() <= 0:
                    continue
                await loc.wait_for(state="visible", timeout=320)
                try:
                    await loc.scroll_into_view_if_needed(timeout=500)
                except Exception:
                    pass
                try:
                    await loc.focus()
                except Exception:
                    pass
                try:
                    await loc.click(timeout=550)
                except Exception:
                    # 某些层级会挡住普通 click，使用 force 再试
                    try:
                        await loc.click(timeout=550, force=True)
                    except Exception:
                        # 部分 Google challenge 元素对键盘事件更敏感
                        try:
                            await loc.press("Enter", timeout=400)
                        except Exception:
                            await loc.press("Space", timeout=400)
                try:
                    await page.wait_for_timeout(180)
                except Exception:
                    pass
                # 先判断是否切到了 totp challenge，再等待输入框出现（页面重时需要更久）
                try:
                    cur = (page.url or "").lower()
                    if "/challenge/totp" in cur or "/challenge/ipp" in cur:
                        _print_and_log(f"[RPA][2FA] challenge 已切换: {page.url}")
                except Exception:
                    pass
                if await _wait_for_2fa_code_input_selector(page, timeout_ms=2600):
                    return True
            except Exception:
                continue

        # JS 兜底点击（兼容 Shadow/事件绑定场景）
        try:
            clicked = bool(
                await page.evaluate(
                    """() => {
						const all = Array.from(document.querySelectorAll("div[role='link'][data-action='selectchallenge'], div[jsname='EBHGs'][data-challengetype], div[role='button']"));
						let target = all.find(el => (el.getAttribute('data-challengetype') || '') === '6');
						if (!target) {
							target = all.find(el => {
								const t = (el.textContent || '').toLowerCase();
								return t.includes('google authenticator') || t.includes('verification code') || t.includes('security code');
							});
						}
						if (!target) return false;
						try { target.scrollIntoView({block:'center', inline:'center'}); } catch (e) {}
						try { target.focus && target.focus(); } catch (e) {}
						if (typeof PointerEvent === 'function') {
							target.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, composed:true}));
							target.dispatchEvent(new PointerEvent('pointerup', {bubbles:true, composed:true}));
						}
						target.dispatchEvent(new MouseEvent('mousedown', {bubbles:true}));
						target.dispatchEvent(new MouseEvent('mouseup', {bubbles:true}));
						target.click();
						target.dispatchEvent(new MouseEvent('click', {bubbles:true}));
						target.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', code:'Enter', bubbles:true}));
						target.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter', code:'Enter', bubbles:true}));
						return true;
					}"""
                )
            )
            if clicked:
                try:
                    await page.wait_for_timeout(280)
                except Exception:
                    pass
                if await _wait_for_2fa_code_input_selector(page, timeout_ms=2800):
                    return True
        except Exception:
            pass

        try:
            await page.wait_for_timeout(120)
        except Exception:
            pass

    # 最后再给一次稍长等待，避免已经点击成功但验证码框慢加载时误判失败
    if await _wait_for_2fa_code_input_selector(page, timeout_ms=2400):
        return True
    return bool(await _wait_for_2fa_code_input_selector(page, timeout_ms=650))


async def _wait_for_google_identifier_step(
    page, *, timeout_ms: int = 12000
) -> Optional[str]:
    """等待 Google 邮箱输入步骤出现，返回可用 selector。"""
    deadline = time.time() + (max(1000, int(timeout_ms)) / 1000.0)
    selectors = [
        "input#identifierId",
        "input[name='identifier']",
        "input[type='email'][aria-label='Email or phone']",
        "div.Xb9hP input[name='identifier']",
    ]
    while time.time() < deadline:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await page.locator(sel).count() > 0:
                    await loc.wait_for(state="visible", timeout=800)
                    return sel
            except Exception:
                pass
        try:
            await page.wait_for_timeout(250)
        except Exception:
            pass
    return None


async def _set_input_value_robust(
    page, *, selectors: list[str], text: str, timeout_ms: int = 10000
) -> bool:
    """稳健输入：fill -> 校验 -> 逐字输入 -> JS 回退。"""
    expect = (text or "").strip()
    if not expect:
        return False
    deadline = time.time() + (max(1000, int(timeout_ms)) / 1000.0)

    def _normalize(v: str) -> str:
        return (v or "").strip()

    while time.time() < deadline:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await page.locator(sel).count() <= 0:
                    continue
                await loc.wait_for(state="visible", timeout=1000)
                try:
                    await loc.click(timeout=1000)
                except Exception:
                    pass

                # 1) 优先 fill（在 Google 输入框更稳定）
                try:
                    await loc.fill("")
                    await loc.fill(expect)
                    val = _normalize(await loc.input_value())
                    if val == expect:
                        return True
                except Exception:
                    pass

                # 2) 退化为逐字输入
                try:
                    await _human_type_text(
                        page,
                        sel,
                        expect,
                        min_key_delay_ms=40,
                        max_key_delay_ms=90,
                        pre_delay_s=0.1,
                        post_delay_s=0.1,
                    )
                    val = _normalize(await loc.input_value())
                    if val == expect:
                        return True
                except Exception:
                    pass

                # 3) JS 直接赋值并派发事件（兜底）
                try:
                    ok = bool(
                        await page.evaluate(
                            """(payload) => {
								const { selector, value } = payload || {};
								const el = document.querySelector(selector);
								if (!el) return false;
								el.focus();
								el.value = value;
								el.dispatchEvent(new Event('input', { bubbles: true }));
								el.dispatchEvent(new Event('change', { bubbles: true }));
								return (el.value || '').trim() === (value || '').trim();
							}""",
                            {"selector": sel, "value": expect},
                        )
                    )
                    if ok:
                        return True
                except Exception:
                    pass
            except Exception:
                continue
        try:
            await page.wait_for_timeout(250)
        except Exception:
            pass
    return False


async def _best_effort_google_consent(page, *, options: ValidateOptions) -> None:
    """尽力处理 OAuth 授权同意页。"""
    # Google 同意/授权页的按钮 DOM 比较稳定：
    # - button[jsname="LgbsSe"] 内通常包含 span[jsname="V67aGc"] 作为按钮文字
    # <button ... jsname="LgbsSe"><span jsname="V67aGc">Đăng nhập</span></button>
    # 因此这里优先用结构化选择器，其次再用多语言文案兜底。

    # 速度优先：你给的 DOM 里同意按钮在 #submit_approve_access 下，优先用该锚点快速命中。
    # 注意：这里故意不做太多等待/随机延迟，避免“同意/登录”触发太慢。
    fast_selectors = [
        "#submit_approve_access button[jsname='LgbsSe']",
        "#submit_approve_access button:has(span[jsname='V67aGc'])",
        "#submit_approve_access button:has(span[jsname='V67aGc']:has-text('登录'))",
        "#submit_approve_access button:has-text('登录')",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('登录'))",
        "button:has-text('登录')",
    ]

    consent_selectors = [
        # 结构化：明确指向 Google 的 Material Button
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('Đăng nhập'))",
        # 结构化 + 常见多语言关键词（同一按钮可能只是“继续/允许/登录”等）
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('Allow'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('Continue'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('同意'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('允许'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('继续'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('下一步'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('Next'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('Sign in'))",
        "button[jsname='LgbsSe']:has(span[jsname='V67aGc']:has-text('Authorize'))",
        # 兜底：Playwright 文案匹配
        "button:has-text('Allow')",
        "button:has-text('允许')",
        "button:has-text('同意')",
        "button:has-text('继续')",
        "button:has-text('Continue')",
        "button:has-text('Đăng nhập')",
        "button:has-text('Sign in')",
        "button:has-text('Authorize')",
        "button:has-text('Next')",
        "button:has-text('下一步')",
    ]

    # 快速路径：短超时 + 高频轮询（更快点到“同意/登录”）
    for i in range(12):
        clicked = await _click_first_visible(page, fast_selectors, timeout_ms=700)
        if clicked:
            try:
                await page.wait_for_timeout(250)
            except Exception:
                pass
            return
        try:
            await page.wait_for_timeout(150)
        except Exception:
            pass

    # 多轮 best-effort：Google 页面可能会先跳转/加载权限列表，再出现按钮
    for i in range(5):
        clicked = await _click_first_visible(page, consent_selectors, timeout_ms=900)
        if clicked:
            # 同意按钮点下去要“快”；避免 4-6 秒的随机 human_delay 让触发变慢
            try:
                await page.wait_for_timeout(300)
            except Exception:
                pass
            return
        try:
            await page.wait_for_timeout(250)
        except Exception:
            pass

    # 最后回退：扫描所有 Google Material Button，按 span 文案做关键词命中后点击
    # 说明：不同语言/地区文案差异大，这里用关键词集合尽力覆盖。
    try:
        keywords = [
            "allow",
            "continue",
            "agree",
            "consent",
            "authorize",
            "sign in",
            "login",
            "next",
            "13ng nh9p",  # 越南语
            "13ng nhadp",  # 容错：去掉声调的常见写法
            "d1ng e9ng",  # (保守备用)
            "d1k",  # 避免误杀：仅兜底时使用
            "3f403e343e3b3638424c",  # 俄语: Continue
            "40303740354838424c",  # 俄语: Allow
            "34303b3535",  # 俄语: Next
            "3f40383d4f424c",  # 俄语: Accept
            "4fd0nd0",  # 中文拼音“同意”(极弱兜底)
            "3e3a",  # OK
            "1e1a",  # OK
            "453e403e483e",  # 俄语: ok/good
            "\u540c\u610f",
            "\u5141\u8bb8",
            "\u7ee7\u7eed",
            "\u4e0b\u4e00\u6b65",
        ]
        btns = page.locator("button[jsname='LgbsSe']")
        cnt = await btns.count()
        for idx in range(min(cnt, 12)):
            b = btns.nth(idx)
            try:
                label = await b.locator("span[jsname='V67aGc']").first.inner_text(
                    timeout=180
                )
            except Exception:
                try:
                    label = await b.inner_text(timeout=180)
                except Exception:
                    label = ""
            label_norm = (label or "").strip().lower()
            if not label_norm:
                continue
            if any(k in label_norm for k in keywords):
                try:
                    await b.wait_for(state="visible", timeout=1200)
                    await b.click()
                    try:
                        await page.wait_for_timeout(300)
                    except Exception:
                        pass
                    return
                except Exception:
                    continue
    except Exception:
        # best-effort：不抛出，交给后续 wait_for_url/人工介入判定
        pass

    await _human_delay(options, reason="after_consent_noop")


async def _click_first_visible(page, selectors, *, timeout_ms: int = 2500) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if await loc.count() <= 0:
                continue
            await loc.first.wait_for(state="visible", timeout=timeout_ms)
            await loc.first.click()
            return True
        except Exception:
            continue
    return False


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="账号池 RPA 验活（Antigravity OAuth + 凭证落盘）"
    )
    p.add_argument("--username", default="", help="Google 登录账号（通常是邮箱）")
    p.add_argument("--password", default="", help="Google 登录密码")
    p.add_argument(
        "--user-session", default="rpa-cli", help="用于隔离 OAuth 流程的会话标识"
    )
    p.add_argument("--headless", action="store_true", help="无头模式")
    p.add_argument(
        "--manual", action="store_true", help="允许人工介入（遇到验证码/2FA 时可用）"
    )
    p.add_argument(
        "--timeout", type=int, default=300, help="超时秒数（等待进入目标页面）"
    )
    p.add_argument("--slowmo", type=int, default=0, help="每步延迟毫秒（调试用）")
    p.add_argument("--locale", default="zh-CN", help="浏览器语言环境，如 zh-CN")
    p.add_argument("--timezone", default="Asia/Shanghai", help="时区，如 Asia/Shanghai")
    p.add_argument("--ua", default="", help="自定义 User-Agent（留空使用默认）")
    p.add_argument("--viewport-width", type=int, default=1366, help="视口宽度")
    p.add_argument("--viewport-height", type=int, default=768, help="视口高度")
    p.add_argument(
        "--human-delay",
        action="store_true",
        help="启用随机延迟（默认 2-4 秒，用于模拟人工停顿/页面节奏）",
    )
    p.add_argument(
        "--human-delay-min", type=float, default=0.0, help="随机延迟最小秒数"
    )
    p.add_argument(
        "--human-delay-max", type=float, default=0.0, help="随机延迟最大秒数"
    )
    p.add_argument(
        "--external-browser",
        action="store_true",
        help="使用系统默认浏览器完成 OAuth（推荐：避免 Playwright 打开 Google 登录页触发风控）",
    )
    p.add_argument(
        "--bitbrowser",
        action="store_true",
        help="使用 BitBrowser 本地服务创建/打开窗口并通过 CDP 连接（createBrowser->openBrowser->connect_over_cdp）",
    )
    p.add_argument(
        "--bitbrowser-id",
        default="",
        help="指定 BitBrowser 窗口 ID（不指定则会调用 createBrowser() 自动创建）",
    )
    p.add_argument(
        "--bitbrowser-auto-delete",
        action="store_true",
        default=True,
        help="流程结束自动 deleteBrowser(id)（仅当使用 BitBrowser 时生效，默认开启）",
    )
    # 提供反向开关以便显式禁用自动删除
    p.add_argument(
        "--no-bitbrowser-auto-delete",
        dest="bitbrowser_auto_delete",
        action="store_false",
        help="禁用流程结束时自动 deleteBrowser(id)",
    )
    return p


async def _amain(argv: list[str]) -> int:
    args = _build_arg_parser().parse_args(argv)

    username = (args.username or "").strip()
    password = (args.password or "").strip()
    if not username or not password:
        # 不传参时给默认值，便于你直接 python main.py 快速测通流程。
        # 注：真实跑请显式传参或设置环境变量 RPA_TEST_USERNAME/RPA_TEST_PASSWORD。
        log.warning(
            "[RPA] 未提供 --username/--password，将使用 TEST 默认值进行测试。"
            "（测试 OK 后请注释/删除 main.py 里的 TEST_DEFAULT_*）"
        )
        username = username or str(TEST_DEFAULT_USERNAME)
        password = password or str(TEST_DEFAULT_PASSWORD)

    # human-delay 快捷开关：默认 2-4 秒
    hd_min = float(args.human_delay_min)
    hd_max = float(args.human_delay_max)
    if bool(args.human_delay):
        hd_min, hd_max = 2.0, 4.0

    # BitBrowser 默认窗口：便于本地测试。
    # - CLI 未传 --bitbrowser-id 时才会使用 TEST_DEFAULT_BITBROWSER_ID。
    # - 置空/注释 TEST_DEFAULT_BITBROWSER_ID 即可禁用该“测试默认值”。
    arg_bit_id = (
        str(args.bitbrowser_id).strip() if args.bitbrowser_id is not None else ""
    )
    test_bit_id = _get_test_default_bitbrowser_id() or ""
    bitbrowser_id = arg_bit_id or (test_bit_id or None)
    opts = ValidateOptions(
        headless=bool(args.headless),
        manual=bool(args.manual),
        timeout_sec=int(args.timeout),
        slow_mo_ms=int(args.slowmo),
        user_agent=(str(args.ua).strip() or None),
        locale=str(args.locale),
        timezone_id=str(args.timezone),
        viewport_width=int(args.viewport_width) if args.viewport_width else None,
        viewport_height=int(args.viewport_height) if args.viewport_height else None,
        external_browser=bool(args.external_browser),
        human_delay_min_sec=hd_min,
        human_delay_max_sec=hd_max,
        # 默认 True：即便 CLI 不传 --bitbrowser 也走 BitBrowser
        bitbrowser=bool(args.bitbrowser) or True,
        bitbrowser_id=bitbrowser_id,
        bitbrowser_auto_delete=bool(args.bitbrowser_auto_delete),
    )

    try:
        result = await validate_antigravity_account(
            username=username,
            password=password,
            user_session=str(args.user_session),
            options=opts,
        )
    except PlaywrightNotInstalled as e:
        log.error(str(e))
        return 2
    except Exception as e:
        # 带堆栈，便于定位 Playwright/BitBrowser 侧异常。
        try:
            log.exception(f"验活失败: {type(e).__name__}: {e}")
        except Exception:
            log.error(f"验活失败: {type(e).__name__}: {e}")
        return 1

    if result.get("success"):
        log.info("验活成功：凭证已保存")
        return 0

    log.error(f"验活失败：{result.get('error')}")
    return 1


def main() -> int:
    # Windows 下 asyncio 默认事件循环策略 OK；这里不强行修改。
    try:
        import asyncio

        return asyncio.run(_amain(sys.argv[1:]))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
