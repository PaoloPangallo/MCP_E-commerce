from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent_stream import router as agent_stream_router
from app.api.routes import router as search_router
from app.api.seller import seller_router
from app.auth.auth_router import router as auth_router
from app.db.database import Base, SessionLocal, engine
from app.mcp.client import MCPToolClient
from app.mcp.server import configure_mcp, mcp

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP E-Commerce API")

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

# 1. Configura l'MCP Tool Context (iniettando il session maker)
configure_mcp(db_factory=SessionLocal)

# 2. Monta le route dell'MCP server.
# Essendo il client (MCPToolClient) configurato per usare streamable_http_client 
# sull'URL '/mcp', espongo l'app streamable_http_app fornita da FastMCP.
app.mount("/mcp", mcp.streamable_http_app)

app.include_router(search_router)
app.include_router(seller_router)
app.include_router(auth_router)
app.include_router(agent_stream_router)

@app.get("/health")
def health():
    return {"status": "ok"}