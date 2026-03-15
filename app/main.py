from __future__ import annotations

import logging

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

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    import asyncio
    from app.services.model_singleton import preload as _preload_model

    # 1) Warm up the SentenceTransformer on the main thread before serving requests
    logger.info("Pre-loading SentenceTransformer model...")
    await asyncio.to_thread(_preload_model)
    logger.info("SentenceTransformer model ready.")

    # 2) Check Redis Connectivity for session memory
    from app.db.redis import redis_client
    try:
        # redis_client is a RedisManager, we need to check its internal state or just try an operation
        if redis_client.get_json("health_check") is None:
            logger.info("Redis cache ready (checked via RedisManager).")
    except Exception as e:
        logger.warning("Could not connect to Redis: %s. Session memory/history will fallback to local memory.", e)

    async with mcp_app.router.lifespan_context(app):
        yield

    logger.info("App shutdown complete.")

app = FastAPI(title="MCP E-Commerce API", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
logger.info("Database tables initialized")

logger.info("Mounting MCP Server at /mcp")
app.mount("/mcp", mcp_app)

app.include_router(search_router)
app.include_router(seller_router)
app.include_router(auth_router)
app.include_router(agent_stream_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}