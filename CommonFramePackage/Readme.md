# CommonFramePackage

可复用的通用运行时工具包，适合多个 Python 项目共享。

## 包含内容

- `hot_reload.py`
  - 热重载监听工具（依赖 `watchdog`）。
- `runtime_bootstrap.py`
  - 运行时目录引导，统一 `project_root`、`CREDENTIALS_DIR`、`APP_BASE_DIR/shared`。
  - 支持自动识别 `.../releases/<version>/` 目录结构并将共享目录定位到 `<AppBase>/shared`。
  - 可选自动切换工作目录到入口脚本所在目录，避免发布目录切换导致相对路径失效。
- `fastapi_keepalive.py`
  - 为 FastAPI 注册 `GET/HEAD /keepalive` 的通用方法。
- `proxy_pool/`
  - 代理池后端可移植模块（SQLite 持久化 + FastAPI 路由 + 批量导入 + 测试 + 绑定）。

## 依赖安装

`hot_reload.py` 依赖 `watchdog`：

```bash
pip install watchdog
```

`proxy_pool/` 额外依赖：

```bash
pip install aiosqlite httpx
```

## 典型用法

```python
from CommonFramePackage.runtime_bootstrap import bootstrap_runtime_context

runtime = bootstrap_runtime_context(__file__)
print(runtime.project_root)
print(runtime.credentials_dir)
```
