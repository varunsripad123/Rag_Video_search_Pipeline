"""Authentication utilities."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import AppConfig, load_config

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


def get_config() -> AppConfig:
    return load_config()


def verify_api_key(api_key: str | None = Security(API_KEY_HEADER), config: AppConfig = Depends(get_config)) -> str:
    if api_key is None or api_key not in set(config.security.api_keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key
