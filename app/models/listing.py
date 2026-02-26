# app/models/listing.py

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.db.database import Base


class Listing(Base):
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True, index=True)

    ebay_id = Column(String, index=True)
    title = Column(String)
    price = Column(Float)
    currency = Column(String)
    condition = Column(String)

    seller_name = Column(String)
    seller_rating = Column(Float)

    url = Column(String)
    image_url = Column(String)

    created_at = Column(DateTime(timezone=True), server_default=func.now())