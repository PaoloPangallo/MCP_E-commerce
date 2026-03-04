from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.seller import seller_router
from app.db.database import Base, engine
from app.models import listing

app = FastAPI()

# 🔥 CORS SUPER PERMISSIVO (debug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB
Base.metadata.create_all(bind=engine)

# routers
app.include_router(router)
app.include_router(seller_router)