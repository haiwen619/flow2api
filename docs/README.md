# Flow2API 使用说明

本文档基于当前项目代码整理（更新时间：2026-03-03），用于快速部署、配置和使用 Flow2API。

## 1. 项目简介

Flow2API 是一个面向 Google Flow/Gemini 能力的 OpenAI 兼容网关，提供：

- OpenAI 风格接口：`/v1/models`、`/v1/chat/completions`
- 图片/视频生成统一入口（支持 `messages` 和 `contents` 两种请求风格）
- Token 管理后台（AT 刷新、Cookie reAuth 刷新、批量导入导出、日志）
- 账号池自动化与代理池管理
- 自动任务：每分钟巡检活跃 Token 的 AT 刷新窗口
- 自动任务：每小时自动解禁因 429 被禁用且满足条件的 Token
- 一次账号导入后全自动化管理（登录态维护、Token 刷新、失效恢复）

## 2. 快速启动

### 2.1 本地启动（Python）

前置要求：

- Python 3.11
- 可联网环境（调用上游接口）
- 如使用浏览器打码/自动登录：安装 Playwright 浏览器内核

步骤：

```bash
# 1) 复制配置（Windows PowerShell）
Copy-Item config/setting_example.toml config/setting.toml
# macOS/Linux 可使用：cp config/setting_example.toml config/setting.toml

# 2) 安装依赖
pip install -r requirements.txt

# 3) (可选) 安装 Playwright 浏览器
playwright install chromium

# 4) 启动服务
python main.py
```

默认监听：

- `http://127.0.0.1:8000`
- 管理台：`http://127.0.0.1:8000/login`

### 2.2 Docker 启动

```bash
# 直接使用远程镜像
docker compose up -d
```

说明：

- `docker-compose.yml`：拉取 `ghcr.io/thesmallhancat/flow2api:latest`
- `docker-compose.local.yml`：本地构建镜像
- `docker-compose.proxy.yml`：附带 WARP 代理容器
- 默认映射端口：`38000 -> 8000`

## 3. 配置说明（config/setting.toml）

首次启动会读取 `setting.toml` 初始化数据库配置；后续也可在管理台修改（数据库配置优先）。

关键配置项：

- `[global].api_key`：外部 API 鉴权密钥（`Authorization: Bearer <api_key>`）
- `[global].admin_username` / `[global].admin_password`：管理台账号密码
- `[server].host` / `[server].port`：服务监听地址和端口
- `[flow].labs_base_url` / `[flow].api_base_url`：上游地址
- `[flow].enable_reauth_refresh`：AT 刷新失败时是否启用 reAuth 恢复链路
- `[flow].reauth_cookie_invalid_auto_login_enabled`：reAuth 命中 `interaction_required` 时，是否触发账号池自动登录恢复
- `[proxy].proxy_enabled` / `[proxy].proxy_url`：请求代理开关和地址
- `[generation].image_timeout` / `[generation].video_timeout`：生成超时
- `[captcha].captcha_method`：`browser` / `personal` / 第三方打码服务
- `[captcha]` 下各服务 API Key 与 Base URL：YesCaptcha / CapMonster / EzCaptcha / CapSolver

## 4. 管理后台与鉴权

### 4.1 外部 API 鉴权

- 头部：`Authorization: Bearer <global.api_key>`
- 使用接口：`/v1/models`、`/v1/chat/completions`

### 4.2 管理后台鉴权

- 登录接口：`POST /api/admin/login`
- 登录成功后返回 session token
- 后续后台接口通过 `Authorization: Bearer <admin_session_token>` 调用
- 账号池与代理池接口也复用同一后台 token 鉴权

## 5. OpenAI 兼容接口使用

### 5.1 查询模型

```bash
curl -X GET "http://127.0.0.1:8000/v1/models" \
  -H "Authorization: Bearer <API_KEY>"
```

### 5.2 统一生成接口

接口：`POST /v1/chat/completions`

支持：

- `messages`（OpenAI 风格）
- `contents`（Gemini 风格）
- `stream: true`（SSE 输出，结尾包含 `data: [DONE]`）
- `stream: false`（等待生成完成后一次性返回 JSON）

### 图片生成示例（gemini-3.1-flash-image）

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "contents": [
      { "parts": [ { "text": "A realistic food photo, studio light, clean table." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "9:16", "imageSize": "1K" }
    },
    "stream": true
  }'
```

### 非流式示例（兼容未接入 SSE 的上游）

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image-landscape",
    "messages": [
      {
        "role": "user",
        "content": "一只可爱的猫咪在花园里玩耍"
      }
    ],
    "stream": false
  }'
```

### 5.3 imageConfig 与模型别名规则

当前代码对 `gemini-3.0-pro-image` 与 `gemini-3.1-flash-image` 采用同一套规则：

- `aspectRatio`：`16:9 -> landscape`，`9:16 -> portrait`，`1:1 -> square`，`4:3 -> four-three`，`3:4 -> three-four`
- `aspectRatio=21:9`：明确不支持（返回 400）
- `imageSize=1K`：默认尺寸（不加 `-2k/-4k` 后缀）
- `imageSize=2K`：模型后缀 `-2k`
- `imageSize=4K`：模型后缀 `-4k`

别名示例（会自动归一化）：

- `gemini-3.1-flash-image-4x3`
- `gemini-3.1-flash-image-4k-16x9`
- `gemini-3.1-flash-image-2k-9x16`
- `gemini-3.1-flash-image-1k`

建议优先调用基础 family（如 `gemini-3.1-flash-image`）并通过 `generationConfig.imageConfig` 控制比例与分辨率。

## 6. Token 管理与自动刷新机制

### 6.1 自动刷新基本规则

- AT 自动刷新开关在当前版本固定启用（接口保留但不支持关闭）
- 每分钟巡检活跃 Token
- 仅在“未过期且剩余 < 1 小时”进入自动刷新窗口
- 已过期 Token 在纯定时巡检中会跳过

### 6.2 近过期刷新策略

近过期时优先执行 reAuth-only 刷新 Cookie/会话：

- 成功：刷新记录标记成功，并更新 Cookie/ST/AT 相关信息
- 失败且命中 `interaction_required`：判定 Cookie 失效
- Token 自动标记失效并停用（`ban_reason=cookie_invalid_need_relogin`）
- 后续自动刷新会跳过该记录
- 前端状态展示为：`Cookie失效(需重新自动登录)`

### 6.3 Cookie 失效触发账号池自动登录

开关：管理页“`Cookie失效自动登录`”（接口：`/api/token-refresh/enabled`）

- 开启后，如果 reAuth 命中 `interaction_required`，系统会按当前 Token 邮箱在账号池检索账号
- 匹配到账号则触发一次自动登录任务（状态会记录为执行中/PENDING）
- 未匹配到账号则仅标记失效，等待人工处理

### 6.4 手动刷新行为

- `刷新AT`：走 AT 刷新链路（必要时包含 fallback）
- `刷Cookie`：仅走 reAuth-only 链路
- 若 Token 之前是 Cookie 失效状态，且 reAuth 手动刷新成功，会自动恢复为活跃状态

## 7. 账号池与代理池

主要接口前缀：

- 账号池：`/accountpool/*`
- 代理池：`/proxypool/*`

页面入口：

- 账号池页面：`/account_pool_page_v2_full`
- 代理池页面：`/proxy_pool_page`

两者均使用后台登录后的 session token 鉴权。

## 8. 数据与日志位置

- SQLite 数据库：`data/flow.db`
- 临时缓存/文件：`tmp/`
- 请求日志：管理台“请求日志”页或接口 `/api/logs`

## 9. 常见问题

- `401 Invalid API key`：检查 `Authorization` 头与 `[global].api_key` 是否一致
- `Unsupported imageConfig.aspectRatio: 21:9`：当前版本不支持 21:9，请改用 16:9 / 9:16 / 1:1 / 4:3 / 3:4
- reAuth 日志出现 `interaction_required`：说明当前 Cookie/登录态已失效；请执行自动登录恢复，或手动重新登录后更新 Token
- 浏览器相关功能不可用：确认已安装 Playwright 依赖与浏览器内核（`playwright install chromium`）

## 10. 升级建议

升级前建议先备份：

- `data/`（数据库与任务数据）
- `config/setting.toml`（基础配置）

升级后优先做三项验证：

- 后台可正常登录（`/login`）
- `/v1/models` 可返回模型列表
- 随机抽一条 Token 执行一次 `刷新AT` 与 `刷Cookie` 验证链路

## 11. 图片生成比例与 imageConfig 速查

这部分对应 [routes.py](/h:/katu/Github/flow2api/src/api/routes.py) 里的 `_normalize_ratio_suffix_from_model()`、`_normalize_aspect_ratio()` 与 `_resolve_model_from_image_config()` 逻辑。

### 11.1 当前支持的比例

| 传入写法 | 标准后缀 | 实际模型示例 |
| --- | --- | --- |
| `16:9` / `16x9` / `landscape` / `image-aspect-ratio-landscape` | `landscape` | `gemini-3.1-flash-image-landscape` |
| `9:16` / `9x16` / `portrait` / `image-aspect-ratio-portrait` | `portrait` | `gemini-3.1-flash-image-portrait` |
| `1:1` / `1x1` / `square` / `image-aspect-ratio-square` | `square` | `gemini-3.1-flash-image-square` |
| `4:3` / `4x3` / `four-three` / `image-aspect-ratio-landscape-four-three` | `four-three` | `gemini-3.1-flash-image-four-three` |
| `3:4` / `3x4` / `three-four` / `image-aspect-ratio-portrait-three-four` | `three-four` | `gemini-3.1-flash-image-three-four` |

不支持的比例：

- `21:9` / `21x9` 会直接返回 `400`
- 其他未在上表中的值也会返回 `400`

### 11.2 imageConfig.aspectRatio 的处理规则

- 入口字段是 `generationConfig.imageConfig.aspectRatio`
- 会先归一化成标准模型后缀：`landscape / portrait / square / four-three / three-four`
- 如果传的是基础 family 模型，例如 `gemini-3.1-flash-image`，系统会自动拼成对应具体模型
- 如果目标具体模型不存在，会自动回退到当前 family 下最接近的可用模型

请求示例：

```json
{
  "model": "gemini-3.1-flash-image",
  "generationConfig": {
    "responseModalities": ["IMAGE"],
    "imageConfig": {
      "aspectRatio": "4:3"
    }
  }
}
```

实际会映射到类似：

```text
gemini-3.1-flash-image-four-three
```

### 11.3 imageConfig.imageSize 的处理规则

支持的写法：

- `1K` / `1k` / `1024` -> 不加尺寸后缀，等价默认尺寸
- `2K` / `2k` / `2048` -> `-2k`
- `4K` / `4k` / `4096` -> `-4k`

不支持的值会返回 `400`。

### 11.4 默认行为

- 如果 `generationConfig` 不存在，且传入的是基础 family 模型，如 `gemini-3.1-flash-image`，默认会落到 `-landscape`
- 如果 `imageConfig` 不是对象，行为同上
- 如果 `imageConfig` 里既没传 `aspectRatio`，也没传 `imageSize`，基础 family 仍默认走 `-landscape`
- 如果只传了 `imageSize`，没传 `aspectRatio`，则优先继承当前模型自带的比例；继承不到时默认 `landscape`
- 如果只传了 `aspectRatio`，没传 `imageSize`，则保留当前模型原本的尺寸后缀；基础尺寸不会额外加 `-1k`

### 11.5 模型别名也支持这些比例写法

除了 `imageConfig.aspectRatio`，模型名后缀本身也支持同样的比例别名归一化，例如：

- `gemini-3.1-flash-image-4x3`
- `gemini-3.1-flash-image-4k-16x9`
- `gemini-3.1-flash-image-2k-9x16`
- `gemini-3.1-flash-image-1k`

这些别名最终也会被规范到标准模型后缀。

### 11.6 gemini-2.5-flash-image 的比例支持说明

`gemini-2.5-flash-image` 当前在模型表里只有两个原生变体：

- `gemini-2.5-flash-image-landscape`
- `gemini-2.5-flash-image-portrait`

也就是说，精确支持的比例只有：

| 传入比例 | 实际模型 |
| --- | --- |
| `16:9` / `16x9` / `landscape` | `gemini-2.5-flash-image-landscape` |
| `9:16` / `9x16` / `portrait` | `gemini-2.5-flash-image-portrait` |

对于下面这些比例：

- `1:1` / `square`
- `4:3` / `four-three`
- `3:4` / `three-four`

当前不是 `gemini-2.5-flash-image` 的原生可用模型。若你传的是基础 family `gemini-2.5-flash-image` 再配合 `imageConfig.aspectRatio`，路由层会按回退规则自动改到最近可用模型：

- `square` 会优先回退到 `landscape`，其次 `portrait`
- `4:3` 会优先回退到 `landscape`
- `3:4` 会优先回退到 `portrait`

因此：

- 想要精确比例输出时，`gemini-2.5-flash-image` 目前只建议用 `16:9` 和 `9:16`
- 如果你需要原生 `1:1` / `4:3` / `3:4`，应优先使用 `gemini-3.0-pro-image` 或 `gemini-3.1-flash-image`
