from fastapi import FastAPI
from app.api.routes import router
from app.api.seller import seller_router
from app.db.database import engine
from app.models import listing  # importante per registrare il modello
from app.db.database import Base
from fastapi.middleware.cors import CORSMiddleware



Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="MCP E-Commerce Backend",
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

app.include_router(router)
app.include_router(seller_router)
