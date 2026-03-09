# app/models/chat.py

import json
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True) # UUID or friendly name
    title = Column(String, nullable=False)
    user_id = Column(Integer, index=True, nullable=True) # Per futura multi-utenza
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), index=True)
    role = Column(String) # user, assistant
    content = Column(Text) # Testo del messaggio
    
    # Dati strutturati (es. risultati ricerca, thinking trace) salvati come JSON
    payload = Column(Text, nullable=True) 
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "payload": json.loads(self.payload) if self.payload else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
