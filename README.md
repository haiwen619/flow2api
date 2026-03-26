# Flow2API

服务器的 MySQL ： Esc_j6qIXg_k1xTgeYL0WLIhKNYfzpYXTsMJq


远程打码填入
https://rt.lmmllm.com
Esc_j6qIXg_k1xTgeYL0WLIhKNYfzpYXTsMJq


New-NetFirewallRule -DisplayName "Sub2API-CRS2-8020" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8020


New-NetFirewallRule -DisplayName "Flow2API-RemoteBrowser-8060" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8060

New-NetFirewallRule -DisplayName "Flow2API-8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000

New-NetFirewallRule -DisplayName "CAP-8137" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8137

http://23.159.248.139:8000/manage
jHrrRDxVD5twXN2t

newapi.apishop.cc
https://newapi.apishop.cc/


& C:\Flow2API\.venv\Scripts\Activate.ps1


https://docs.cqtai.com/nano%E7%94%9F%E6%88%90/

Linux  1panl面板  
uvicorn src.main:app --host 0.0.0.0 --port 8000


# 1) 基础模型 + imageConfig: 4:3 + 1K
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image",
    "contents": [
      { "parts": [ { "text": "A realistic food photo, studio light, clean table." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "21:9", "imageSize": "1K" }
    },
    "stream": true
  }'

# 1) 基础模型 + imageConfig: 4:3 + 1K 服务器地址示例
curl -X POST "http://23.159.248.139:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image",
    "contents": [
      { "parts": [ { "text": "A realistic food photo, studio light, clean table." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "1:1", "imageSize": "1K" }
    },
    "stream": true
  }'


# ---------------------------------------------------------start
$body = @'
{
  "model": "gemini-3.0-pro-image",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "生成一个牛年大吉的图片给我，场景是雪地，然后注意要把细节处理好，减少AI生成感觉，然后可以加一些其他小动物在旁边"
        }
      ]
    }
  ],
  "generationConfig": {
    "responseModalities": ["IMAGE"],
    "imageConfig": { "aspectRatio": "1:1", "imageSize": "1K" }
  },
  "stream": true
}
'@

curl.exe -X POST "http://23.159.248.139:3000/v1/chat/completions" `
  -H "Authorization: Bearer sk-lPSOlrLXS6KfFq12yDdXa4d3cc9Bcx5BatP9Lf9mdVTPDFAf" `
  -H "Content-Type: application/json" `
  -d $body

#  ----------------------------------------------------------


# 2) 基础模型 + imageConfig: 3:4 + 2K
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image",
    "contents": [
      { "parts": [ { "text": "生成一只可爱小狗" } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "9:16", "imageSize": "1K" }
    },
    "stream": true
  }'
# 3) 基础模型 + imageConfig: 1:1 + 4K
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image",
    "contents": [
      { "parts": [ { "text": "A minimal logo mascot, flat design, white background." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "1:1", "imageSize": "4K" }
    },
    "stream": true
  }'
# 4) 别名模型写法: 4x3（等价 four-three，默认 1K）
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image-4x3",
    "messages": [
      { "role": "user", "content": "A cozy cafe interior, warm sunlight, film look." }
    ],
    "stream": true
  }'

# 5) 别名模型写法: 3x4 + 2k
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image-3x4-2k",
    "messages": [
      { "role": "user", "content": "A fashion editorial portrait, soft shadows, detailed skin texture." }
    ],
    "stream": true
  }'

  https://limited-sky-yukon-deer.trycloudflare.com/


# 6) 别名模型写法: 1x1 + 4k
curl -X POST "https://limited-sky-yukon-deer.trycloudflare.com/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image-1x1-1k",
    "messages": [
      { "role": "user", "content": "A cute 3D toy character, centered composition, high detail." }
    ],
    "stream": true
  }'

# 7) 测试 2.5 模型 + imageConfig（会被 imageConfig 覆盖为 16:9 + 1K） 
curl -X POST "https://limited-sky-yukon-deer.trycloudflare.com/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image",
    "contents": [
      { "parts": [ { "text": "A realistic food photo, studio light, clean table." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "4:3", "imageSize": "1K" }
    },
    "stream": true
  }'




curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.0-pro-image",
    "contents": [
      { "parts": [ { "text": "A realistic food photo, studio light, clean table." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "4:3", "imageSize": "1k" }
    },
    "stream": true
  }'



curl -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer sk-lPSOlrLXS6KfFq12yDdXa4d3cc9Bcx5BatP9Lf9mdVTPDFAf" \
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

curl -X POST "http://23.159.248.139:3000/v1/chat/completions" \
  -H "Authorization: Bearer sk-lPSOlrLXS6KfFq12yDdXa4d3cc9Bcx5BatP9Lf9mdVTPDFAf" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "contents": [
      { "parts": [ { "text": "A realistic food photo, studio light, clean table." } ] }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": { "aspectRatio": "16:9", "imageSize": "1K" }
    },
    "stream": true
  }'




# ---------------------------------------------------------start
$body = @'
{
  "model": "gemini-3.1-flash-image",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "生成一个牛年大吉的图片给我，场景是雪地，然后注意要把细节处理好，减少AI生成感觉，然后可以加一些其他小动物在旁边"
        }
      ]
    }
  ],
  "generationConfig": {
    "responseModalities": ["IMAGE"],
    "imageConfig": { "aspectRatio": "16:9", "imageSize": "1K" }
  },
  "stream": true
}
'@

curl.exe -X POST "http://127.0.0.1:3000/v1/chat/completions" `
  -H "Authorization: Bearer sk-lPSOlrLXS6KfFq12yDdXa4d3cc9Bcx5BatP9Lf9mdVTPDFAf" `
  -H "Content-Type: application/json" `
  -d $body

#  ----------------------------------------------------------



curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
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




curl -X POST "http://23.159.248.139:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
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






curl -X POST "http://38.49.55.206:8000/v1/chat/completions" \
  -H "Authorization: Bearer jHrrRDxVD5twXN2t" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image-landscape",
    "messages": [
      {
        "role": "user",
        "content": "一只可爱的猫咪在花园里玩耍"
      }
    ],
  "generationConfig": {
    "responseModalities": ["IMAGE"],
    "imageConfig": { "aspectRatio": "16:9", "imageSize": "1K" }
  },
  "stream": true
  }'














<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.119.0-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

**一个功能完整的 OpenAI 兼容 API 服务，为 Flow 提供统一的接口**

</div>

## ✨ 核心特性

- 🎨 **文生图** / **图生图**
- 🎬 **文生视频** / **图生视频**
- 🎞️ **首尾帧视频**
- 🔄 **AT/ST自动刷新** - AT 过期自动刷新，ST 过期时自动通过浏览器更新（personal 模式）
- 📊 **余额显示** - 实时查询和显示 VideoFX Credits
- 🚀 **负载均衡** - 多 Token 轮询和并发控制
- 🌐 **代理支持** - 支持 HTTP/SOCKS5 代理
- 📱 **Web 管理界面** - 直观的 Token 和配置管理
- 🎨 **图片生成连续对话**
- 🧩 **Gemini 官方请求体兼容** - 支持 `generateContent` / `streamGenerateContent`、`systemInstruction`、`contents.parts.text/inlineData/fileData`
- ✅ **Gemini 官方格式已实测出图** - 已使用真实 Token 验证 `/models/{model}:generateContent` 可正常返回官方 `candidates[].content.parts[].inlineData`

## 🚀 快速开始

### 前置要求

- Docker 和 Docker Compose（推荐）
- 或 Python 3.11

> 本地/Linux 直接运行时，推荐固定使用 `Python 3.11`。  
> 当前不要使用 `Python 3.14` 执行 `pip install -r requirements.txt`，否则会在 `pydantic-core`、`curl-cffi` 等依赖上触发源码编译并初始化失败。

- 由于Flow增加了额外的验证码，你可以自行选择使用浏览器打码或第三发打码：
注册[YesCaptcha](https://yescaptcha.com/i/13Xd8K)并获取api key，将其填入系统配置页面```YesCaptcha API密钥```区域
- 默认 `docker-compose.yml` 建议搭配第三方打码（yescaptcha/capmonster/ezcaptcha/capsolver）。
如需 Docker 内有头打码（browser/personal），请使用下方 `docker-compose.headed.yml`。

- 自动更新st浏览器拓展：[Flow2API-Token-Updater](https://github.com/TheSmallHanCat/Flow2API-Token-Updater)

### 方式一：Docker 部署（推荐）

#### 标准模式（不使用代理）

```bash
# 克隆项目
git clone https://github.com/TheSmallHanCat/flow2api.git
cd flow2api

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

> 说明：Compose 已默认挂载 `./tmp:/app/tmp`。如果把缓存超时设为 `0`，语义是“不自动过期删除”；若希望容器重建后仍保留缓存文件，也需要保留这个 `tmp` 挂载。

#### WARP 模式（使用代理）

```bash
# 使用 WARP 代理启动
docker-compose -f docker-compose.warp.yml up -d

# 查看日志
docker-compose -f docker-compose.warp.yml logs -f
```

#### Docker 有头打码模式（browser / personal）

> 适用于你有虚拟化桌面需求、希望在容器里启用有头浏览器打码的场景。  
> 该模式默认启动 `Xvfb + Fluxbox` 实现容器内部可视化，并设置 `ALLOW_DOCKER_HEADED_CAPTCHA=true`。  
> 仅开放应用端口，不提供任何远程桌面连接端口。

```bash
# 启动有头模式（首次建议带 --build）
docker compose -f docker-compose.headed.yml up -d --build

# 查看日志
docker compose -f docker-compose.headed.yml logs -f
```

- API 端口：`8000`
- 进入管理后台后，将验证码方式设为 `browser` 或 `personal`

### 方式二：本地部署

```bash
# 克隆项目
git clone https://github.com/TheSmallHanCat/flow2api.git
cd flow2api

# 复制配置
cp config/setting_example.toml config/setting.toml

# Linux 推荐显式使用 Python 3.11
python3.11 -m venv .venv

# 激活虚拟环境
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 升级 pip
python -m pip install -U pip

# 安装依赖
pip install -r requirements.txt

# 默认 setting_example.toml 的 captcha_method=browser
# 首次本地启动建议安装 Chromium
python -m playwright install chromium

# 启动服务
python main.py
```

如果你的 Linux 服务器已经在用 `python3`/`pip`，先确认版本：

```bash
python3 --version
pip3 --version
```

若输出是 `Python 3.14.x`，请改成下面这种方式初始化：

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
python -m playwright install chromium
python main.py
```

### 首次访问

服务启动后,访问管理后台: **http://localhost:8000**,首次登录后请立即修改密码!

- **用户名**: `admin`
- **密码**: `admin`

## 📋 支持的模型

### 图片生成

| 模型名称 | 说明| 尺寸 |
|---------|--------|--------|
| `gemini-2.5-flash-image-landscape` | 图/文生图 | 横屏 |
| `gemini-2.5-flash-image-portrait` | 图/文生图 | 竖屏 |
| `gemini-3.0-pro-image-landscape` | 图/文生图 | 横屏 |
| `gemini-3.0-pro-image-portrait` | 图/文生图 | 竖屏 |
| `gemini-3.0-pro-image-square` | 图/文生图 | 方图 |
| `gemini-3.0-pro-image-four-three` | 图/文生图 | 横屏 4:3 |
| `gemini-3.0-pro-image-three-four` | 图/文生图 | 竖屏 3:4 |
| `gemini-3.0-pro-image-landscape-2k` | 图/文生图(2K) | 横屏 |
| `gemini-3.0-pro-image-portrait-2k` | 图/文生图(2K) | 竖屏 |
| `gemini-3.0-pro-image-square-2k` | 图/文生图(2K) | 方图 |
| `gemini-3.0-pro-image-four-three-2k` | 图/文生图(2K) | 横屏 4:3 |
| `gemini-3.0-pro-image-three-four-2k` | 图/文生图(2K) | 竖屏 3:4 |
| `gemini-3.0-pro-image-landscape-4k` | 图/文生图(4K) | 横屏 |
| `gemini-3.0-pro-image-portrait-4k` | 图/文生图(4K) | 竖屏 |
| `gemini-3.0-pro-image-square-4k` | 图/文生图(4K) | 方图 |
| `gemini-3.0-pro-image-four-three-4k` | 图/文生图(4K) | 横屏 4:3 |
| `gemini-3.0-pro-image-three-four-4k` | 图/文生图(4K) | 竖屏 3:4 |
| `imagen-4.0-generate-preview-landscape` | 图/文生图 | 横屏 |
| `imagen-4.0-generate-preview-portrait` | 图/文生图 | 竖屏 |
| `gemini-3.1-flash-image-landscape` | 图/文生图 | 横屏 |
| `gemini-3.1-flash-image-portrait` | 图/文生图 | 竖屏 |
| `gemini-3.1-flash-image-square` | 图/文生图 | 方图 |
| `gemini-3.1-flash-image-four-three` | 图/文生图 | 横屏 4:3 |
| `gemini-3.1-flash-image-three-four` | 图/文生图 | 竖屏 3:4 |
| `gemini-3.1-flash-image-landscape-2k` | 图/文生图(2K) | 横屏 |
| `gemini-3.1-flash-image-portrait-2k` | 图/文生图(2K) | 竖屏 |
| `gemini-3.1-flash-image-square-2k` | 图/文生图(2K) | 方图 |
| `gemini-3.1-flash-image-four-three-2k` | 图/文生图(2K) | 横屏 4:3 |
| `gemini-3.1-flash-image-three-four-2k` | 图/文生图(2K) | 竖屏 3:4 |
| `gemini-3.1-flash-image-landscape-4k` | 图/文生图(4K) | 横屏 |
| `gemini-3.1-flash-image-portrait-4k` | 图/文生图(4K) | 竖屏 |
| `gemini-3.1-flash-image-square-4k` | 图/文生图(4K) | 方图 |
| `gemini-3.1-flash-image-four-three-4k` | 图/文生图(4K) | 横屏 4:3 |
| `gemini-3.1-flash-image-three-four-4k` | 图/文生图(4K) | 竖屏 3:4 |

说明：

- `gemini-2.5-flash-image` 当前只原生支持 `landscape(16:9)` 和 `portrait(9:16)`
- 如果通过 `imageConfig.aspectRatio` 传入 `1:1` / `4:3` / `3:4`，路由层只会自动回退到最近可用的横屏或竖屏，不代表 `2.5` 原生支持这些比例
- 需要原生 `square / four-three / three-four` 时，优先使用 `gemini-3.0-pro-image` 或 `gemini-3.1-flash-image`

### 视频生成

#### 文生视频 (T2V - Text to Video)
⚠️ **不支持上传图片**

| 模型名称 | 说明| 尺寸 |
|---------|---------|--------|
| `veo_3_1_t2v_fast_portrait` | 文生视频 | 竖屏 |
| `veo_3_1_t2v_fast_landscape` | 文生视频 | 横屏 |
| `veo_2_1_fast_d_15_t2v_portrait` | 文生视频 | 竖屏 |
| `veo_2_1_fast_d_15_t2v_landscape` | 文生视频 | 横屏 |
| `veo_2_0_t2v_portrait` | 文生视频 | 竖屏 |
| `veo_2_0_t2v_landscape` | 文生视频 | 横屏 |
| `veo_3_1_t2v_fast_portrait_ultra` | 文生视频 | 竖屏 |
| `veo_3_1_t2v_fast_ultra` | 文生视频 | 横屏 |
| `veo_3_1_t2v_fast_portrait_ultra_relaxed` | 文生视频 | 竖屏 |
| `veo_3_1_t2v_fast_ultra_relaxed` | 文生视频 | 横屏 |
| `veo_3_1_t2v_portrait` | 文生视频 | 竖屏 |
| `veo_3_1_t2v_landscape` | 文生视频 | 横屏 |

#### 首尾帧模型 (I2V - Image to Video)
📸 **支持1-2张图片：1张作为首帧，2张作为首尾帧**

> 💡 **自动适配**：系统会根据图片数量自动选择对应的 model_key
> - **单帧模式**（1张图）：使用首帧生成视频
> - **双帧模式**（2张图）：使用首帧+尾帧生成过渡视频

| 模型名称 | 说明| 尺寸 |
|---------|---------|--------|
| `veo_3_1_i2v_s_fast_portrait_fl` | 图生视频 | 竖屏 |
| `veo_3_1_i2v_s_fast_fl` | 图生视频 | 横屏 |
| `veo_2_1_fast_d_15_i2v_portrait` | 图生视频 | 竖屏 |
| `veo_2_1_fast_d_15_i2v_landscape` | 图生视频 | 横屏 |
| `veo_2_0_i2v_portrait` | 图生视频 | 竖屏 |
| `veo_2_0_i2v_landscape` | 图生视频 | 横屏 |
| `veo_3_1_i2v_s_fast_portrait_ultra_fl` | 图生视频 | 竖屏 |
| `veo_3_1_i2v_s_fast_ultra_fl` | 图生视频 | 横屏 |
| `veo_3_1_i2v_s_fast_portrait_ultra_relaxed` | 图生视频 | 竖屏 |
| `veo_3_1_i2v_s_fast_ultra_relaxed` | 图生视频 | 横屏 |
| `veo_3_1_i2v_s_portrait` | 图生视频 | 竖屏 |
| `veo_3_1_i2v_s_landscape` | 图生视频 | 横屏 |

#### 多图生成 (R2V - Reference Images to Video)
🖼️ **支持多张图片**

> **2026-03-06 更新**
>
> - 已同步上游新版 `R2V` 视频请求体
> - `textInput` 已切换为 `structuredPrompt.parts`
> - 顶层新增 `mediaGenerationContext.batchId`
> - 顶层新增 `useV2ModelConfig: true`
> - 横屏 / 竖屏 `R2V` 模型共用同一套新版请求体
> - 横屏 `R2V` 的上游 `videoModelKey` 已切换为 `*_landscape` 形式
> - 根据当前上游协议，`referenceImages` 当前最多传 **3 张**

| 模型名称 | 说明| 尺寸 |
|---------|---------|--------|
| `veo_3_1_r2v_fast_portrait` | 图生视频 | 竖屏 |
| `veo_3_1_r2v_fast` | 图生视频 | 横屏 |
| `veo_3_1_r2v_fast_portrait_ultra` | 图生视频 | 竖屏 |
| `veo_3_1_r2v_fast_ultra` | 图生视频 | 横屏 |
| `veo_3_1_r2v_fast_portrait_ultra_relaxed` | 图生视频 | 竖屏 |
| `veo_3_1_r2v_fast_ultra_relaxed` | 图生视频 | 横屏 |

#### 视频放大模型 (Upsample)

| 模型名称 | 说明 | 输出 |
|---------|---------|--------|
| `veo_3_1_t2v_fast_portrait_4k` | 文生视频放大 | 4K |
| `veo_3_1_t2v_fast_4k` | 文生视频放大 | 4K |
| `veo_3_1_t2v_fast_portrait_ultra_4k` | 文生视频放大 | 4K |
| `veo_3_1_t2v_fast_ultra_4k` | 文生视频放大 | 4K |
| `veo_3_1_t2v_fast_portrait_1080p` | 文生视频放大 | 1080P |
| `veo_3_1_t2v_fast_1080p` | 文生视频放大 | 1080P |
| `veo_3_1_t2v_fast_portrait_ultra_1080p` | 文生视频放大 | 1080P |
| `veo_3_1_t2v_fast_ultra_1080p` | 文生视频放大 | 1080P |
| `veo_3_1_i2v_s_fast_portrait_ultra_fl_4k` | 图生视频放大 | 4K |
| `veo_3_1_i2v_s_fast_ultra_fl_4k` | 图生视频放大 | 4K |
| `veo_3_1_i2v_s_fast_portrait_ultra_fl_1080p` | 图生视频放大 | 1080P |
| `veo_3_1_i2v_s_fast_ultra_fl_1080p` | 图生视频放大 | 1080P |
| `veo_3_1_r2v_fast_portrait_ultra_4k` | 多图视频放大 | 4K |
| `veo_3_1_r2v_fast_ultra_4k` | 多图视频放大 | 4K |
| `veo_3_1_r2v_fast_portrait_ultra_1080p` | 多图视频放大 | 1080P |
| `veo_3_1_r2v_fast_ultra_1080p` | 多图视频放大 | 1080P |

## 📡 API 使用示例（需要使用流式）

> 除了下方 `OpenAI-compatible` 示例，服务也支持 Gemini 官方格式：
> - `POST /v1beta/models/{model}:generateContent`
> - `POST /models/{model}:generateContent`
> - `POST /v1beta/models/{model}:streamGenerateContent`
> - `POST /models/{model}:streamGenerateContent`
>
> Gemini 官方格式支持以下认证方式：
> - `Authorization: Bearer <api_key>`
> - `x-goog-api-key: <api_key>`
> - `?key=<api_key>`
>
> Gemini 官方图片请求体已兼容：
> - `systemInstruction`
> - `contents[].parts[].text`
> - `contents[].parts[].inlineData`
> - `contents[].parts[].fileData.fileUri`
> - `generationConfig.responseModalities`
> - `generationConfig.imageConfig.aspectRatio`
> - `generationConfig.imageConfig.imageSize`

### Gemini 官方 generateContent（文生图）

> 已使用真实 Token 实测通过。
> 如需流式返回，可将路径替换为 `:streamGenerateContent?alt=sse`。

```bash
curl -X POST "http://localhost:8000/models/gemini-3.1-flash-image:generateContent" \
  -H "x-goog-api-key: han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "systemInstruction": {
      "parts": [
        {
          "text": "Return an image only."
        }
      ]
    },
    "contents": [
      {
        "role": "user",
        "parts": [
          {
            "text": "一颗放在木桌上的红苹果，棚拍光线，极简背景"
          }
        ]
      }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": {
        "aspectRatio": "1:1",
        "imageSize": "1K"
      }
    }
  }'
```

### 文生图

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image-landscape",
    "messages": [
      {
        "role": "user",
        "content": "一只可爱的猫咪在花园里玩耍"
      }
    ],
    "stream": true
  }'
```

非流式兼容调用：

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
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

### 图生图

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image-landscape",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "将这张图片变成水彩画风格"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,<base64_encoded_image>"
            }
          }
        ]
      }
    ],
    "stream": true
  }'
```

### 文生视频

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo_3_1_t2v_fast_landscape",
    "messages": [
      {
        "role": "user",
        "content": "一只小猫在草地上追逐蝴蝶"
      }
    ],
    "stream": true
  }'
```

### 首尾帧生成视频

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo_3_1_i2v_s_fast_fl_landscape",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "从第一张图过渡到第二张图"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,<首帧base64>"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,<尾帧base64>"
            }
          }
        ]
      }
    ],
    "stream": true
  }'
```

### 多图生成视频

> `R2V` 会由服务端自动组装新版视频请求体，调用方仍然使用 OpenAI 兼容输入即可。
> 服务端会将横屏 `R2V` 自动映射到最新的 `*_landscape` 上游模型键。
> 当前最多传 **3 张参考图**。

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo_3_1_r2v_fast_portrait",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "以三张参考图的人物和场景为基础，生成一段镜头平滑推进的竖屏视频"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64/<参考图1base64>"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64/<参考图2base64>"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64/<参考图3base64>"
            }
          }
        ]
      }
    ],
    "stream": true
  }'
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [PearNoDec](https://github.com/PearNoDec) 提供的YesCaptcha打码方案
- [raomaiping](https://github.com/raomaiping) 提供的无头打码方案
感谢所有贡献者和使用者的支持！

---

## 📞 联系方式

- 提交 Issue：[GitHub Issues](https://github.com/TheSmallHanCat/flow2api/issues)

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TheSmallHanCat/flow2api&type=date&legend=top-left)](https://www.star-history.com/#TheSmallHanCat/flow2api&type=date&legend=top-left)
