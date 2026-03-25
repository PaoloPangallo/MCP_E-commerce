from __future__ import annotations

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ENV: str = "development"
    EBAY_USER_TOKEN: str = ""
    EBAY_CLIENT_ID: str = ""
    EBAY_CLIENT_SECRET: str = ""
    EBAY_ENV: str = "sandbox"
    EBAY_MARKETPLACE_ID: str = "EBAY_IT"
    MCP_SERVER_URL: str = "http://127.0.0.1:8050/mcp"
    NGROK_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"

    # Search config
    EBAY_PAGE_SIZE: int = 20
    EBAY_MAX_OFFSET_PAGES: int = 3
    EBAY_REQUEST_TIMEOUT: float = 15.0
    APPROX_PRICE_PCT: float = 0.2
    APPROX_PRICE_MIN_DELTA: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() in {"production", "prod"}

settings = Settings()
