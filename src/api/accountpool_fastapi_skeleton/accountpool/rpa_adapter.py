from __future__ import annotations

from typing import Any, Dict


async def validate_account_via_rpa(
    *,
    username: str,
    password: str,
    job_id: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Replace this function with your real RPA implementation.

    Expected return schema:
    {
      "success": bool,
      "message": str | None,
      "error": str | None,
      "file_path": str | None,
      "auto_detected_project": bool | None
    }
    """
    # TODO: integrate BrowserAutomation + OAuth + credential persistence.
    # This placeholder keeps API contract stable for front-end integration.
    if not username or not password:
        return {"success": False, "error": "missing username/password"}
    return {
        "success": True,
        "message": "rpa_stub_success",
        "error": None,
        "file_path": None,
        "auto_detected_project": None,
    }

