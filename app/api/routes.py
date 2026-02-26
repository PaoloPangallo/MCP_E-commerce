from fastapi import APIRouter
from pydantic import BaseModel
from app.services.parser import parse_query_service

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/parse")
def parse_query(request: QueryRequest):
    return parse_query_service(request.query)
