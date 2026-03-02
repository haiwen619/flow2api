"""FastAPI router factory for proxy pool endpoints."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from .models import (
    ProxyPoolAddProxyRequest,
    ProxyPoolBindRequest,
    ProxyPoolBulkImportRequest,
    ProxyPoolDisableRequest,
    ProxyPoolTestRequest,
    ProxyPoolUpdateProxyRequest,
)
from .service import ProxyPoolService

TokenVerifier = Callable[..., Awaitable[str]]


async def _noop_verify_token() -> str:
    return ""


def create_proxy_pool_router(
    service: ProxyPoolService,
    *,
    verify_token: Optional[TokenVerifier] = None,
) -> APIRouter:
    token_dependency = verify_token or _noop_verify_token
    router = APIRouter(dependencies=[Depends(token_dependency)])

    @router.post("/proxypool/proxies")
    async def add_proxy(request: ProxyPoolAddProxyRequest) -> JSONResponse:
        try:
            item = await service.add_proxy(
                host=request.host,
                port=request.port,
                username=request.username,
                password=request.password,
                tags=request.tags or [],
            )
            return JSONResponse(content={"success": True, "item": item})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/proxypool/proxies")
    async def list_proxies(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        search: str = Query("", description="Search host/username/proxy_key"),
        host: str = Query("", description="Exact host filter"),
    ) -> JSONResponse:
        try:
            result = await service.list_proxies(
                offset=offset,
                limit=limit,
                search=search or None,
                host=host or None,
            )
            return JSONResponse(content={"success": True, **result})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.put("/proxypool/proxies/{proxy_key}")
    async def update_proxy(
        proxy_key: str,
        request: ProxyPoolUpdateProxyRequest,
    ) -> JSONResponse:
        try:
            item = await service.update_proxy(
                proxy_key=proxy_key,
                host=request.host,
                port=request.port,
                username=request.username,
                password=request.password,
                tags=request.tags,
            )
            return JSONResponse(content={"success": True, "item": item})
        except KeyError:
            raise HTTPException(status_code=404, detail="proxy not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.delete("/proxypool/proxies/{proxy_key}")
    async def delete_proxy(proxy_key: str) -> JSONResponse:
        try:
            await service.delete_proxy(proxy_key=proxy_key)
            return JSONResponse(content={"success": True})
        except KeyError:
            raise HTTPException(status_code=404, detail="proxy not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/proxypool/proxies/bulk")
    async def bulk_import(request: ProxyPoolBulkImportRequest) -> JSONResponse:
        try:
            result = await service.bulk_import(
                text=request.text,
                default_scheme=request.default_scheme or "http",
                tags=request.tags or [],
            )
            return JSONResponse(content={"success": True, **result})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/proxypool/proxies/{proxy_key}/test")
    async def test_proxy(
        proxy_key: str,
        request: ProxyPoolTestRequest,
    ) -> JSONResponse:
        try:
            result = await service.test_proxy(
                proxy_key=proxy_key,
                timeout_s=float(request.timeout_s or 8.0),
            )
            return JSONResponse(content={"success": True, **result})
        except KeyError:
            raise HTTPException(status_code=404, detail="proxy not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/proxypool/proxies/{proxy_key}/bind")
    async def bind_proxy(
        proxy_key: str,
        request: ProxyPoolBindRequest,
    ) -> JSONResponse:
        try:
            item = await service.bind_proxy(
                proxy_key=proxy_key,
                credential_name=request.credential_name,
                mode=request.mode or "antigravity",
                force=bool(request.force),
            )
            return JSONResponse(content={"success": True, "item": item})
        except KeyError:
            raise HTTPException(status_code=404, detail="proxy not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/proxypool/proxies/{proxy_key}/unbind")
    async def unbind_proxy(proxy_key: str) -> JSONResponse:
        try:
            item = await service.unbind_proxy(proxy_key=proxy_key)
            return JSONResponse(content={"success": True, "item": item})
        except KeyError:
            raise HTTPException(status_code=404, detail="proxy not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/proxypool/proxies/{proxy_key}/disable")
    async def disable_proxy(
        proxy_key: str,
        request: ProxyPoolDisableRequest,
    ) -> JSONResponse:
        try:
            item = await service.set_disabled(
                proxy_key=proxy_key,
                disabled=bool(request.disabled),
            )
            return JSONResponse(content={"success": True, "item": item})
        except KeyError:
            raise HTTPException(status_code=404, detail="proxy not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return router
