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
        String(2000), # JSON encoded {category:budget, brand:budget}
        nullable=True
    )

    # -------------------------
    # Auto-learned Profiling (retrocompatible, server_default=NULL)
    # -------------------------
    category_affinities = Column(
        String(2000),  # JSON: {"Cellulari & smartphone": 12, "Scarpe": 5}
        nullable=True,
        server_default=None,
    )

    condition_preference = Column(
        String(20),    # "new" | "used" | "refurbished" | null
        nullable=True,
        server_default=None,
    )

    interaction_depth = Column(
        String(20),    # "browser" | "researcher" | "power_buyer"
        nullable=True,
        server_default=None,
    )

    # -------------------------
    # Metadata
    # -------------------------
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )