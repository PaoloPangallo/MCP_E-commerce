from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.db.database import Base


class WishlistItem(Base):
    __tablename__ = "wishlist_items"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to the user
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # eBay product data
    ebay_id = Column(String(100), nullable=False)
    title = Column(String(500), nullable=True)
    price = Column(Float, nullable=True)
    previous_price = Column(Float, nullable=True)  # Stores last price before a drop
    currency = Column(String(10), nullable=True, default="EUR")
    condition = Column(String(100), nullable=True)
    image_url = Column(String(1000), nullable=True)
    url = Column(String(1000), nullable=True)
    seller_name = Column(String(255), nullable=True)

    # Metadata
    added_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_checked_at = Column(DateTime(timezone=True), nullable=True)  # Last price check timestamp
