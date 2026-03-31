```



git fetch upstream
git merge upstream/main

先备份并确保工作区干净
git status
git branch backup/main-before-upstream-20260302
添加上游仓库（只需一次）
git remote add upstream https://github.com/TheSmallHanCat/flow2api.git
git remote -v

拉取上游最新代码
git fetch upstream

同步到你的 main（推荐用 merge，简单稳妥）
git checkout main
git pull --ff-only origin main
git merge upstream/main


待办：多账号同时自动化登录时可能有问题，
需要考虑多账号同时登录时，如何处理账号切换的问题，比如如何判断当前登录的账号，如何切换账号等。


Linux 部署 


sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv

cd /path/to/flow2api
cp config/setting_example.toml config/setting.toml

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# 默认配置是 browser 打码，首次建议安装 Chromium
python -m playwright install chromium

python main.py




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
      "imageConfig": { "aspectRatio": "16:9", "imageSize": "2K" }
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







