from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AccountPoolAddAccountRequest(BaseModel):
    platform: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    uid: Optional[str] = None
    tags: Optional[List[str]] = None


class AccountPoolUpdateAccountRequest(BaseModel):
    platform: Optional[str] = None
    display_name: Optional[str] = None
    password: Optional[str] = None
    uid: Optional[str] = None
    tags: Optional[List[str]] = None


class AccountPoolBatchValidateRequest(BaseModel):
    account_keys: List[str] = Field(..., min_length=1)


class AccountPoolItem(BaseModel):
    id: int
    account_key: str
    platform: str
    display_name: str
    uid: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    last_validate_at: Optional[float] = None
    last_validate_ok: Optional[int] = None
    last_validate_status: Optional[str] = None
    last_validate_error: Optional[str] = None
    last_validate_job_id: Optional[str] = None
    last_validate_msg: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class AccountPoolListResponse(BaseModel):
    total: int
    items: List[AccountPoolItem]
    offset: int
    limit: int


class ValidationResult(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    auto_detected_project: Optional[bool] = None


class ValidationJobSafeView(BaseModel):
    job_id: str
    account_key: str
    username: Optional[str] = None
    status: str
    created_at: float
    updated_at: float
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

