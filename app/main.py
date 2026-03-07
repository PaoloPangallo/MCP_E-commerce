from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as search_router
from app.api.seller import seller_router
from app.api.agent_stream import router as agent_stream_router
from app.auth.auth_router import router as auth_router

from app.db.database import Base, engine


app = FastAPI(title="MCP E-Commerce API")


# -----------------------------
# CORS
# -----------------------------

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


# -----------------------------
# Database
# -----------------------------

Base.metadata.create_all(bind=engine)


# -----------------------------
# Routers
# -----------------------------

app.include_router(search_router)
app.include_router(seller_router)
app.include_router(auth_router)
app.include_router(agent_stream_router)


# -----------------------------
# Health check
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}