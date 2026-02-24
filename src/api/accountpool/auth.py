from fastapi import Depends

from ..admin import verify_admin_token


async def verify_panel_token(
    token: str = Depends(verify_admin_token),
) -> str:
    """Reuse current admin session-token auth for account-pool endpoints."""
    return token
