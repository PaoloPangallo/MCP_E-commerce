from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent_stream import router as agent_stream_router
from app.api.routes import router as search_router
from app.api.seller import seller_router
from app.auth.auth_router import router as auth_router
from app.db.database import Base, engine

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from app.mcp.asgi import app as mcp_app

_ENV = os.getenv("ENV", "development").strip().lower()
_IS_PROD = _ENV in {"production", "prod"}

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    import asyncio
    from app.services.model_singleton import preload as _preload_model

    logger.info("Pre-loading SentenceTransformer model...")
    await asyncio.to_thread(_preload_model)
    logger.info("SentenceTransformer model ready.")

    from app.db.redis import redis_client
    try:
        if redis_client.get_json("health_check") is None:
            logger.info("Redis cache ready (checked via RedisManager).")
    except Exception as e:
        logger.warning("Could not connect to Redis: %s. Session memory/history will fallback to local memory.", e)

    from app.services import ebay
    await ebay.init_http_client()
    logger.info("eBay shared HTTP client initialized.")

    async with mcp_app.router.lifespan_context(app):
        yield

    await ebay.close_http_client()
    logger.info("App shutdown complete.")


# Build CORS origins from environment
_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
_frontend_urls = os.getenv("FRONTEND_URLS", "").strip()
if _frontend_urls:
    # Support multiple comma-separated URLs
    origins = [u.strip() for u in _frontend_urls.split(",") if u.strip()]
    _allowed_origins.extend(origins)
    for u in origins:
        logger.info("CORS: Added remote frontend origin: %s", u)


app = FastAPI(
    title="MCP E-Commerce API",
    lifespan=app_lifespan,
    # Disable interactive docs in production
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
    openapi_url=None if _IS_PROD else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger.info("Mounting MCP Server at /mcp")
app.mount("/mcp", mcp_app)

app.include_router(search_router)
app.include_router(seller_router)
app.include_router(auth_router)
app.include_router(agent_stream_router)


@app.get("/")
def root():
    return {
        "message": "MCP E-Commerce API is running",
        "health_check": "/health",
        "documentation": "Disabled in production" if _IS_PROD else "/docs"
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}