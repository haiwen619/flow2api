# Proxy Pool Package

可移植的代理池后端模块，包含：

- SQLite 持久化：代理 CRUD、启用/禁用、绑定/解绑、测试结果回写
- FastAPI 路由：`/proxypool/*` 全套接口
- 批量导入解析：支持 `host:port:user:pass` 和 `host:.. port:.. username:.. password:..`
- 单代理可用性测试：通过 `httpx` 验证出口可达并提取 IP
- 业务辅助：按凭证一对一绑定分配代理、获取可用代理 URL

## 快速接入

```python
from fastapi import FastAPI

from CommonFramePackage.proxy_pool import (
    ProxyPoolRepository,
    ProxyPoolService,
    create_proxy_pool_router,
)

repo = ProxyPoolRepository(db_path="./creds/credentials.db")
service = ProxyPoolService(repo)

app = FastAPI()

@app.on_event("startup")
async def _startup():
    await service.initialize()

app.include_router(
    create_proxy_pool_router(service, verify_token=verify_panel_token),  # 可选
    prefix="",
    tags=["ProxyPool"],
)
```

## 依赖

- `fastapi`
- `pydantic`
- `httpx`
- `aiosqlite`



