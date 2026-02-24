"""
Minimal mount example for target FastAPI project.

Run:
  uvicorn app_mount_example:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from accountpool.auth import get_panel_password
from accountpool.repository import AccountPoolRepository
from accountpool.router import create_accountpool_router
from accountpool.service import AccountPoolService

repo = AccountPoolRepository()
service = AccountPoolService(repo)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.initialize()
    yield


app = FastAPI(title="Target Project - AccountPool Skeleton", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_accountpool_router(service), prefix="", tags=["AccountPool"])


class LoginRequest(BaseModel):
    password: str


@app.post("/auth/login")
async def login(request: LoginRequest):
    # Keep compatibility with existing front-end:
    # account_pool_page_v2_full.html reads token from localStorage key:
    # `gcli2api_auth_token`
    if request.password == get_panel_password():
        return {"token": request.password, "message": "login success"}
    return {"detail": "password mismatch"}


front_dir = Path(__file__).resolve().parent / "front"
if front_dir.exists():
    app.mount("/front", StaticFiles(directory=str(front_dir)), name="front")
else:
    # In real target project, point this to your front root folder.
    pass

