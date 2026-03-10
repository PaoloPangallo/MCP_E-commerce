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
    async with mcp_app.router.lifespan_context(app):
        yield

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