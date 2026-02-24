from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .repository import AccountPoolRepository
from .rpa_adapter import validate_account_via_rpa


class AccountPoolService:
    def __init__(self, repo: AccountPoolRepository) -> None:
        self.repo = repo
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.batch_tasks: Dict[str, Dict[str, Any]] = {}
        self.max_jobs = 1000

    async def initialize(self) -> None:
        await self.repo.initialize()

    async def add_account(
        self,
        *,
        platform: str,
        display_name: str,
        password: str,
        uid: Optional[str],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return await self.repo.upsert_account(
            platform=platform,
            display_name=display_name,
            password=password,
            uid=uid,
            tags=tags,
        )

    async def list_accounts(
        self,
        *,
        offset: int,
        limit: int,
        search: Optional[str],
        platform: Optional[str],
    ) -> Dict[str, Any]:
        return await self.repo.list_accounts(offset=offset, limit=limit, search=search, platform=platform)

    async def update_account(
        self,
        *,
        account_key: str,
        platform: Optional[str],
        display_name: Optional[str],
        password: Optional[str],
        uid: Optional[str],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        return await self.repo.update_account(
            account_key=account_key,
            platform=platform,
            display_name=display_name,
            password=password,
            uid=uid,
            tags=tags,
        )

    async def delete_account(self, *, account_key: str) -> None:
        await self.repo.delete_account(account_key=account_key)

    def _default_concurrency(self) -> int:
        try:
            v = int(os.getenv("ACCOUNTPOOL_VALIDATE_CONCURRENCY_DEFAULT", "5") or "5")
        except Exception:
            v = 5
        return max(1, v)

    def _max_concurrency(self) -> int:
        try:
            v = int(os.getenv("ACCOUNTPOOL_VALIDATE_CONCURRENCY_MAX", "20") or "20")
        except Exception:
            v = 20
        return max(1, v)

    def clamp_concurrency(self, value: Optional[int]) -> int:
        if value is None:
            return min(self._default_concurrency(), self._max_concurrency())
        try:
            v = int(value)
        except Exception:
            return min(self._default_concurrency(), self._max_concurrency())
        return min(max(1, v), self._max_concurrency())

    def _prune_jobs(self) -> None:
        if len(self.jobs) <= self.max_jobs:
            return
        ordered = sorted(self.jobs.items(), key=lambda kv: kv[1].get("created_at", 0))
        to_delete = max(1, len(self.jobs) - self.max_jobs)
        for job_id, _ in ordered[:to_delete]:
            self.jobs.pop(job_id, None)

    async def trigger_single_validate(
        self,
        *,
        account_key: str,
        params: Dict[str, Any],
    ) -> str:
        secret = await self.repo.get_account_secret(account_key=account_key)
        display_name = (secret.get("display_name") or "").strip()
        uid = (secret.get("uid") or "").strip()
        password = secret.get("password")
        username = uid if ("@" in uid and len(uid) >= 5) else display_name

        if not username:
            raise ValueError("account missing username")
        if password is None or not str(password).strip():
            raise ValueError("account missing password")

        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {
            "job_id": job_id,
            "account_key": account_key,
            "username": username,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "error": None,
            "result": None,
            "cancelled": False,
            "batch_task_id": None,
        }
        self._prune_jobs()
        asyncio.create_task(
            self._run_single_job(
                job_id=job_id,
                account_key=account_key,
                username=username,
                password=str(password),
                params=params,
                batch_task_id=None,
            )
        )
        return job_id

    async def _run_single_job(
        self,
        *,
        job_id: str,
        account_key: str,
        username: str,
        password: str,
        params: Dict[str, Any],
        batch_task_id: Optional[str],
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        if job.get("cancelled"):
            job["status"] = "cancelled"
            job["updated_at"] = time.time()
            return

        job["status"] = "running"
        job["updated_at"] = time.time()
        await self.repo.set_validation(
            account_key=account_key,
            status="running",
            ok=None,
            error=None,
            job_id=job_id,
            message=None,
            set_validate_at=False,
        )

        try:
            result = await validate_account_via_rpa(
                username=username,
                password=password,
                job_id=job_id,
                params=params,
            )
            ok = bool(result.get("success"))
            job["result"] = result
            job["status"] = "success" if ok else "failed"
            job["error"] = None if ok else (result.get("error") or "unknown error")
            job["updated_at"] = time.time()

            await self.repo.set_validation(
                account_key=account_key,
                status="success" if ok else "failed",
                ok=ok,
                error=(None if ok else job["error"]),
                job_id=job_id,
                message=(result.get("message") if isinstance(result, dict) else None),
                set_validate_at=True,
            )
            self._mark_batch_progress(batch_task_id=batch_task_id, ok=ok)
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["updated_at"] = time.time()
            await self.repo.set_validation(
                account_key=account_key,
                status="failed",
                ok=False,
                error=str(e),
                job_id=job_id,
                message=None,
                set_validate_at=True,
            )
            self._mark_batch_progress(batch_task_id=batch_task_id, ok=False)

    def get_job_safe(self, *, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError("job not found")
        safe = {
            "job_id": job.get("job_id"),
            "account_key": job.get("account_key"),
            "username": job.get("username"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "error": job.get("error"),
        }
        if job.get("status") in ("success", "failed"):
            result = job.get("result")
            if isinstance(result, dict):
                safe["result"] = {
                    "success": bool(result.get("success")),
                    "file_path": result.get("file_path"),
                    "auto_detected_project": result.get("auto_detected_project"),
                    "message": result.get("message"),
                }
        return safe

    async def create_batch_task(
        self,
        *,
        account_keys: List[str],
        concurrency: Optional[int],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        keys = [str(x).strip() for x in account_keys if str(x).strip()]
        if not keys:
            raise ValueError("account_keys is empty")

        batch_id = uuid.uuid4().hex
        self.batch_tasks[batch_id] = {
            "batch_task_id": batch_id,
            "account_keys": keys,
            "job_ids": [],
            "total": len(keys),
            "completed": 0,
            "success": 0,
            "failed": 0,
            "status": "pending",
            "error": None,
            "cancelled": False,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        jobs: List[Dict[str, str]] = []
        for key in keys:
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "job_id": job_id,
                "account_key": key,
                "username": "",
                "status": "queued",
                "created_at": time.time(),
                "updated_at": time.time(),
                "error": None,
                "result": None,
                "cancelled": False,
                "batch_task_id": batch_id,
            }
            self.batch_tasks[batch_id]["job_ids"].append(job_id)
            jobs.append({"account_key": key, "job_id": job_id})

        eff_conc = self.clamp_concurrency(concurrency)
        asyncio.create_task(self._run_batch_task(batch_task_id=batch_id, jobs=jobs, concurrency=eff_conc, params=params))
        return {
            "batch_task_id": batch_id,
            "total": len(keys),
            "concurrency": eff_conc,
        }

    async def _run_batch_task(
        self,
        *,
        batch_task_id: str,
        jobs: List[Dict[str, str]],
        concurrency: int,
        params: Dict[str, Any],
    ) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            return
        task["status"] = "running"
        task["updated_at"] = time.time()
        sem = asyncio.Semaphore(concurrency)

        async def run_one(job_info: Dict[str, str], start_delay_sec: float) -> None:
            await asyncio.sleep(max(0.0, float(start_delay_sec)))
            t = self.batch_tasks.get(batch_task_id)
            if not t or t.get("cancelled"):
                return
            job_id = job_info["job_id"]
            account_key = job_info["account_key"]
            async with sem:
                t2 = self.batch_tasks.get(batch_task_id)
                if not t2 or t2.get("cancelled"):
                    return
                try:
                    secret = await self.repo.get_account_secret(account_key=account_key)
                    display_name = (secret.get("display_name") or "").strip()
                    uid = (secret.get("uid") or "").strip()
                    password = secret.get("password")
                    username = uid if ("@" in uid and len(uid) >= 5) else display_name
                    if not username or not password:
                        raise ValueError("account missing username/password")
                    if job_id in self.jobs:
                        self.jobs[job_id]["username"] = username
                    await self._run_single_job(
                        job_id=job_id,
                        account_key=account_key,
                        username=username,
                        password=str(password),
                        params=params,
                        batch_task_id=batch_task_id,
                    )
                except Exception as e:
                    if job_id in self.jobs:
                        self.jobs[job_id]["status"] = "failed"
                        self.jobs[job_id]["error"] = str(e)
                        self.jobs[job_id]["updated_at"] = time.time()
                    self._mark_batch_progress(batch_task_id=batch_task_id, ok=False)

        tasks = [
            asyncio.create_task(run_one(job, idx * 5.0))
            for idx, job in enumerate(jobs)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        final_task = self.batch_tasks.get(batch_task_id)
        if not final_task:
            return
        if final_task.get("cancelled"):
            final_task["status"] = "cancelled"
        elif final_task.get("status") not in ("failed",):
            final_task["status"] = "completed"
        final_task["updated_at"] = time.time()

    def _mark_batch_progress(self, *, batch_task_id: Optional[str], ok: bool) -> None:
        if not batch_task_id:
            return
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            return
        task["completed"] = int(task.get("completed", 0)) + 1
        if ok:
            task["success"] = int(task.get("success", 0)) + 1
        else:
            task["failed"] = int(task.get("failed", 0)) + 1
        task["updated_at"] = time.time()

    def list_batch_tasks(self) -> List[Dict[str, Any]]:
        rows = []
        for _, task in self.batch_tasks.items():
            rows.append(
                {
                    "batch_task_id": task.get("batch_task_id"),
                    "total": task.get("total"),
                    "completed": task.get("completed"),
                    "success": task.get("success"),
                    "failed": task.get("failed"),
                    "status": task.get("status"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at"),
                    "error": task.get("error"),
                    "cancelled": task.get("cancelled"),
                }
            )
        rows.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return rows

    def get_batch_task(self, *, batch_task_id: str) -> Dict[str, Any]:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        return {
            "batch_task_id": task.get("batch_task_id"),
            "total": task.get("total"),
            "completed": task.get("completed"),
            "success": task.get("success"),
            "failed": task.get("failed"),
            "status": task.get("status"),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "error": task.get("error"),
            "cancelled": task.get("cancelled"),
        }

    def delete_batch_task(self, *, batch_task_id: str) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        del self.batch_tasks[batch_task_id]

    def cancel_batch_task(self, *, batch_task_id: str) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        status = str(task.get("status") or "")
        if status in ("completed", "failed", "cancelled"):
            raise ValueError("task already finished")
        task["cancelled"] = True
        task["status"] = "cancelling"
        task["updated_at"] = time.time()
        for job_id in list(task.get("job_ids", [])):
            if job_id in self.jobs:
                self.jobs[job_id]["cancelled"] = True
                self.jobs[job_id]["status"] = "cancelled"
                self.jobs[job_id]["error"] = "task cancelled"
                self.jobs[job_id]["updated_at"] = time.time()

    def list_batch_jobs(self, *, batch_task_id: str) -> List[Dict[str, Any]]:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        rows: List[Dict[str, Any]] = []
        for job_id in task.get("job_ids", []):
            job = self.jobs.get(job_id)
            if not job:
                continue
            rows.append(
                {
                    "job_id": job.get("job_id"),
                    "account_key": job.get("account_key"),
                    "username": job.get("username"),
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "updated_at": job.get("updated_at"),
                    "error": job.get("error"),
                }
            )
        return rows

    def delete_batch_job(self, *, batch_task_id: str, job_id: str) -> None:
        task = self.batch_tasks.get(batch_task_id)
        if not task:
            raise KeyError("batch task not found")
        if job_id not in task.get("job_ids", []):
            raise KeyError("job not found in batch task")
        task["job_ids"] = [x for x in task.get("job_ids", []) if x != job_id]
        task["total"] = max(0, int(task.get("total", 0)) - 1)
        task["updated_at"] = time.time()
        self.jobs.pop(job_id, None)

