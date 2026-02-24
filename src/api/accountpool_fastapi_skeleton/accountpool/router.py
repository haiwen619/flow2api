from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from .auth import verify_panel_token
from .models import (
    AccountPoolAddAccountRequest,
    AccountPoolBatchValidateRequest,
    AccountPoolUpdateAccountRequest,
)
from .service import AccountPoolService


def create_accountpool_router(service: AccountPoolService) -> APIRouter:
    router = APIRouter()

    @router.post("/accountpool/accounts")
    async def add_account(
        request: AccountPoolAddAccountRequest,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            item = await service.add_account(
                platform=request.platform,
                display_name=request.display_name,
                password=request.password,
                uid=request.uid,
                tags=request.tags or [],
            )
            return JSONResponse(content={"success": True, "item": item})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/accountpool/accounts")
    async def list_accounts(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        search: str = Query("", description="Search display_name/uid/platform"),
        platform: str = Query("", description="Exact platform filter"),
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            result = await service.list_accounts(
                offset=offset,
                limit=limit,
                search=search or None,
                platform=platform or None,
            )
            return JSONResponse(content={"success": True, **result})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/accountpool/accounts/{account_key}")
    async def update_account(
        account_key: str,
        request: AccountPoolUpdateAccountRequest,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            item = await service.update_account(
                account_key=account_key,
                platform=request.platform,
                display_name=request.display_name,
                password=request.password,
                uid=request.uid,
                tags=request.tags,
            )
            return JSONResponse(content={"success": True, "item": item})
        except KeyError:
            raise HTTPException(status_code=404, detail="account not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/accountpool/accounts/{account_key}")
    async def delete_account(
        account_key: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            await service.delete_account(account_key=account_key)
            return JSONResponse(content={"success": True})
        except KeyError:
            raise HTTPException(status_code=404, detail="account not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/accountpool/accounts/{account_key}/validate")
    async def validate_single_account(
        account_key: str,
        headless: Optional[bool] = Query(None),
        timeout_sec: int = Query(300, ge=30, le=1200),
        manual: bool = Query(False),
        external_browser: Optional[bool] = Query(None),
        locale: str = Query("zh-CN"),
        timezone: str = Query("Asia/Shanghai"),
        ua: str = Query(""),
        viewport_width: int = Query(1366, ge=320, le=4096),
        viewport_height: int = Query(768, ge=320, le=2160),
        slow_mo_ms: int = Query(0, ge=0, le=2000),
        human_delay: bool = Query(False),
        human_delay_min: float = Query(0.0, ge=0.0, le=30.0),
        human_delay_max: float = Query(0.0, ge=0.0, le=30.0),
        bitbrowser: bool = Query(False),
        bitbrowser_id: str = Query(""),
        bitbrowser_auto_delete: bool = Query(False),
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        params: Dict[str, Any] = {
            "headless": headless,
            "timeout_sec": timeout_sec,
            "manual": manual,
            "external_browser": external_browser,
            "locale": locale,
            "timezone": timezone,
            "ua": ua,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "slow_mo_ms": slow_mo_ms,
            "human_delay": human_delay,
            "human_delay_min": human_delay_min,
            "human_delay_max": human_delay_max,
            "bitbrowser": bitbrowser,
            "bitbrowser_id": bitbrowser_id,
            "bitbrowser_auto_delete": bitbrowser_auto_delete,
        }
        try:
            job_id = await service.trigger_single_validate(account_key=account_key, params=params)
            return JSONResponse(content={"success": True, "job_id": job_id})
        except KeyError:
            raise HTTPException(status_code=404, detail="account not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/accountpool/validate/status/{job_id}")
    async def validate_job_status(
        job_id: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            return JSONResponse(content={"success": True, "job": service.get_job_safe(job_id=job_id)})
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/accountpool/validate/batch/task")
    async def create_batch_task(
        request: AccountPoolBatchValidateRequest,
        concurrency: Optional[int] = Query(None, ge=1, le=50),
        headless: Optional[bool] = Query(None),
        timeout_sec: int = Query(300, ge=30, le=1200),
        manual: bool = Query(False),
        external_browser: Optional[bool] = Query(None),
        locale: str = Query("zh-CN"),
        timezone: str = Query("Asia/Shanghai"),
        ua: str = Query(""),
        viewport_width: int = Query(1366, ge=320, le=4096),
        viewport_height: int = Query(768, ge=320, le=2160),
        slow_mo_ms: int = Query(0, ge=0, le=2000),
        human_delay: bool = Query(False),
        human_delay_min: float = Query(0.0, ge=0.0, le=30.0),
        human_delay_max: float = Query(0.0, ge=0.0, le=30.0),
        bitbrowser: bool = Query(False),
        bitbrowser_id: str = Query(""),
        bitbrowser_auto_delete: bool = Query(False),
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        params: Dict[str, Any] = {
            "headless": headless,
            "timeout_sec": timeout_sec,
            "manual": manual,
            "external_browser": external_browser,
            "locale": locale,
            "timezone": timezone,
            "ua": ua,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "slow_mo_ms": slow_mo_ms,
            "human_delay": human_delay,
            "human_delay_min": human_delay_min,
            "human_delay_max": human_delay_max,
            "bitbrowser": bitbrowser,
            "bitbrowser_id": bitbrowser_id,
            "bitbrowser_auto_delete": bitbrowser_auto_delete,
        }
        try:
            result = await service.create_batch_task(
                account_keys=request.account_keys,
                concurrency=concurrency,
                params=params,
            )
            return JSONResponse(content={"success": True, **result})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/accountpool/validate/batch/tasks")
    async def list_batch_tasks(
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            return JSONResponse(content={"success": True, "tasks": service.list_batch_tasks()})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/accountpool/validate/batch/task/{batch_task_id}")
    async def get_batch_task(
        batch_task_id: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            return JSONResponse(content={"success": True, "task": service.get_batch_task(batch_task_id=batch_task_id)})
        except KeyError:
            raise HTTPException(status_code=404, detail="batch task not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/accountpool/validate/batch/task/{batch_task_id}")
    async def delete_batch_task(
        batch_task_id: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            service.delete_batch_task(batch_task_id=batch_task_id)
            return JSONResponse(content={"success": True, "message": "batch task deleted"})
        except KeyError:
            raise HTTPException(status_code=404, detail="batch task not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/accountpool/validate/batch/task/{batch_task_id}/cancel")
    async def cancel_batch_task(
        batch_task_id: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            service.cancel_batch_task(batch_task_id=batch_task_id)
            return JSONResponse(content={"success": True, "message": "cancel request accepted"})
        except KeyError:
            raise HTTPException(status_code=404, detail="batch task not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/accountpool/validate/batch/task/{batch_task_id}/jobs")
    async def list_batch_jobs(
        batch_task_id: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            return JSONResponse(content={"success": True, "jobs": service.list_batch_jobs(batch_task_id=batch_task_id)})
        except KeyError:
            raise HTTPException(status_code=404, detail="batch task not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/accountpool/validate/batch/task/{batch_task_id}/job/{job_id}")
    async def delete_batch_job(
        batch_task_id: str,
        job_id: str,
        token: str = Depends(verify_panel_token),
    ) -> JSONResponse:
        _ = token
        try:
            service.delete_batch_job(batch_task_id=batch_task_id, job_id=job_id)
            return JSONResponse(content={"success": True, "message": "job deleted"})
        except KeyError:
            raise HTTPException(status_code=404, detail="job or batch task not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router

