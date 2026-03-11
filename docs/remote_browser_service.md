# Remote Browser Service

This repository now includes a deployable FastAPI skeleton for `remote_browser`.

Install dependencies:

```bash
pip install -r requirements.remote_browser.txt
python -m playwright install chromium
```

App entry:

```bash
uvicorn src.remote_browser_service.app:app --host 0.0.0.0 --port 8060

python main.py --host 0.0.0.0 --port 8000

```

Service home page:

```text
http://127.0.0.1:8060/
```

The home page shows:

- service runtime status
- browser slot / concurrency usage
- recent solve/custom-score/finish/error task history
- active sessions
- proxy pool overview

Swagger / OpenAPI:

```text
http://127.0.0.1:8060/docs
http://127.0.0.1:8060/redoc
http://127.0.0.1:8060/openapi.json
```

Swagger authorize:

- Click `Authorize`
- Input `REMOTE_BROWSER_API_KEY`
- Swagger will send it as `Bearer <token>`

Required environment variables:

```bash
REMOTE_BROWSER_API_KEY=change_me
```

Config file alternative:

```text
config/remote_browser_service.toml
```

Example:

```toml
[remote_browser_service]
api_key = "change_me"
```

Priority:

- `REMOTE_BROWSER_API_KEY` environment variable takes precedence
- if env is empty, the service falls back to `config/remote_browser_service.toml`
- you can override the file path with `REMOTE_BROWSER_CONFIG_PATH`

Optional environment variables:

```bash
REMOTE_BROWSER_HOST=0.0.0.0
REMOTE_BROWSER_PORT=8060
REMOTE_BROWSER_BROWSER_COUNT=1
REMOTE_BROWSER_USER_DATA_DIR=browser_data_remote_service
REMOTE_BROWSER_SESSION_TTL_SECONDS=3600
REMOTE_BROWSER_SESSION_REAPER_INTERVAL_SECONDS=60
REMOTE_BROWSER_WARMUP_ON_STARTUP=false
BROWSER_EXECUTABLE_PATH=/path/to/chrome
ALLOW_DOCKER_HEADED_CAPTCHA=true
DISPLAY=:99
```

Docker build:

```bash
docker build -f Dockerfile.remote_browser -t flow2api-remote-browser .
```

Why the image is large:

- `playwright install --with-deps chromium` is the main contributor.
- It downloads Chromium itself and installs a large set of runtime libraries required for headed browser automation.
- This repository now uses `requirements.remote_browser.txt` for the remote-browser image, so it no longer installs unrelated packages such as `nodriver`.
- The Dockerfile also copies only `src/` and `config/` instead of the whole repository, and clears apt/cache files after browser installation.

Proxy behavior:

- The remote-browser service now reads proxy information from the existing proxy-pool data.
- If a request carries `token_id`, the service will try to resolve a proxy for that token from proxy pool bindings first.
- If no token-bound proxy is available, it falls back to any available proxy from proxy pool.
- If system proxy config is enabled in the main Flow2API database, system proxy still has higher priority.

Docker run:

```bash
docker run --rm -p 8060:8060 \
  -e REMOTE_BROWSER_API_KEY=change_me \
  -e REMOTE_BROWSER_BROWSER_COUNT=2 \
  flow2api-remote-browser
```

systemd files included in repo:

```text
deploy/systemd/flow2api-remote-browser.service
deploy/systemd/remote-browser.env.example
```

Nginx reverse-proxy example included in repo:

```text
deploy/nginx/flow2api-remote-browser.conf
```

Endpoints:

- `POST /api/v1/solve`
- `POST /api/v1/sessions/{session_id}/finish`
- `POST /api/v1/sessions/{session_id}/error`
- `POST /api/v1/custom-score`
- `GET /healthz`

Authentication:

- All `/api/v1/*` endpoints require `Authorization: Bearer <REMOTE_BROWSER_API_KEY>`.

Example solve request:

```bash
curl -X POST "http://127.0.0.1:8060/api/v1/solve" \
  -H "Authorization: Bearer change_me" \
  -H "Content-Type: application/json" \
  -d "{\"project_id\":\"your_project_id\",\"action\":\"IMAGE_GENERATION\"}"
```

Example custom-score request:

```bash
curl -X POST "http://127.0.0.1:8060/api/v1/custom-score" \
  -H "Authorization: Bearer change_me" \
  -H "Content-Type: application/json" \
  -d "{\"website_url\":\"https://antcpt.com/score_detector/\",\"website_key\":\"6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf\",\"verify_url\":\"https://antcpt.com/score_detector/verify.php\",\"action\":\"homepage\",\"enterprise\":false}"
```

Flow2API integration:

```toml
[captcha]
captcha_method = "remote_browser"
remote_browser_base_url = "http://127.0.0.1:8060"
remote_browser_api_key = "change_me"
remote_browser_timeout = 60
```

Notes:

- The service reuses the existing `src/services/browser_captcha.py` implementation.
- `finish` and `error` are intentionally idempotent. Unknown `session_id` returns `200` with `found=false`.
- The session registry is only used to map external `session_id` to the internal browser handle returned by the local browser service.
- The root path `/` redirects to Swagger docs.
- Java 调用方接入说明见 [docs/remote_browser_service_java.md](/h:/katu/Github/flow2api/docs/remote_browser_service_java.md)。

## systemd deployment

Assume project path:

```text
/opt/flow2api
```

Create runtime user:

```bash
sudo useradd -r -s /usr/sbin/nologin flow2api
sudo mkdir -p /etc/flow2api
sudo chown -R flow2api:flow2api /opt/flow2api
```

Create environment file:

```bash
sudo cp deploy/systemd/remote-browser.env.example /etc/flow2api/remote-browser.env
sudo nano /etc/flow2api/remote-browser.env
```

At minimum set:

```bash
REMOTE_BROWSER_API_KEY=change_me
REMOTE_BROWSER_HOST=127.0.0.1
REMOTE_BROWSER_PORT=8060
```

Install systemd service:

```bash
sudo cp deploy/systemd/flow2api-remote-browser.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable flow2api-remote-browser
sudo systemctl start flow2api-remote-browser
```

Check status:

```bash
sudo systemctl status flow2api-remote-browser
journalctl -u flow2api-remote-browser -f
```

## Nginx reverse proxy

Copy the sample config:

```bash
sudo cp deploy/nginx/flow2api-remote-browser.conf /etc/nginx/conf.d/
sudo nginx -t
sudo systemctl reload nginx
```

Default upstream in the sample points to:

```text
http://127.0.0.1:8060
```

After that you can expose:

```text
http://remote-browser.example.com/docs
http://remote-browser.example.com/healthz
```

If you use HTTPS, pair this server block with your existing Certbot or TLS config and keep the same upstream target.
