"""Pydantic models for proxy pool APIs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ProxyPoolAddProxyRequest(BaseModel):
    host: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    tags: Optional[List[str]] = None


class ProxyPoolUpdateProxyRequest(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    username: Optional[str] = None
    password: Optional[str] = None
    tags: Optional[List[str]] = None


class ProxyPoolBulkImportRequest(BaseModel):
    text: str = Field(..., min_length=1)
    default_scheme: Optional[str] = "http"
    tags: Optional[List[str]] = None


class ProxyPoolTestRequest(BaseModel):
    timeout_s: Optional[float] = Field(8.0, ge=1.0, le=60.0)


class ProxyPoolBindRequest(BaseModel):
    credential_name: str = Field(..., min_length=1)
    mode: Optional[str] = "antigravity"
    force: Optional[bool] = False


class ProxyPoolDisableRequest(BaseModel):
    disabled: bool = Field(..., description="true=disabled, false=enabled")
