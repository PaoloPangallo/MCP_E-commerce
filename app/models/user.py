from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from app.db.database import Base


class User(Base):

    __tablename__ = "users"

    # -------------------------
    # Primary key
    # -------------------------
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # -------------------------
    # Authentication
    # -------------------------
    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )

    password_hash = Column(
        String(255),
        nullable=False
    )

    # -------------------------
    # Preferences
    # -------------------------
    favorite_brands = Column(
        String(255),
        nullable=True
    )

    price_preference = Column(
        String(50),
        nullable=True
    )

    custom_instructions = Column(
        String(1000),
        nullable=True
    )

    theme = Column(
        String(20),
        default="light",
        server_default="light",
        nullable=False
    )

    conversation_tone = Column(
        String(50),
        default="neutral",
        server_default="neutral",
        nullable=False
    )

    contextual_budgets = Column(
        String(2000), # JSON encoded category/brand budgets
        nullable=True
    )

    # -------------------------
    # Metadata
    # -------------------------
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )