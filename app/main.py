from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent_routes import router
from app.api.seller import seller_router
from app.db.database import Base, engine
from app.auth.auth_router import router as auth_router
from app.models.listing import Listing
from app.models.chat import ChatSession, ChatMessage

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB
Base.metadata.create_all(bind=engine)

# routers
from app.api.chat_routes import router as chat_router

app.include_router(router)
app.include_router(seller_router)
app.include_router(auth_router)
app.include_router(chat_router)