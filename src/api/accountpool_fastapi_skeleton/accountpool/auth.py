import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=True)


def get_panel_password() -> str:
    panel_pwd = os.getenv("PANEL_PASSWORD")
    if panel_pwd:
        return panel_pwd
    fallback = os.getenv("PASSWORD")
    if fallback:
        return fallback
    return "pwd"


async def verify_panel_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    password = get_panel_password()
    if credentials.credentials != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="password mismatch")
    return credentials.credentials

