# app/api/chat_routes.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import uuid

from app.db.database import get_db
from app.models.chat import ChatSession, ChatMessage
from app.auth.dependencies import get_optional_user

router = APIRouter(prefix="/chats", tags=["Chat Management"])

@router.get("/")
def list_chats(db: Session = Depends(get_db), user = Depends(get_optional_user)):
    user_id = user.id if user else None
    chats = db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc()).all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at} for c in chats]

@router.get("/{session_id}")
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return {
        "id": session.id,
        "title": session.title,
        "messages": [m.to_dict() for m in messages]
    }

@router.delete("/{session_id}")
def delete_chat(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    
    db.delete(session)
    db.commit()
    return {"status": "ok"}
