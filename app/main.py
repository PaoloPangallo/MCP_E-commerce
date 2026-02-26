from fastapi import FastAPI
from app.api.routes import router
from app.db.database import engine
from app.models import listing  # importante per registrare il modello
from app.db.database import Base

Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="MCP E-Commerce Backend",
    version="0.1.0"
)

app.include_router(router)
