import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.models.wishlist import WishlistItem

logger = logging.getLogger(__name__)

wishlist_router = APIRouter(prefix="/wishlist", tags=["Wishlist"])


class AddItemRequest(BaseModel):
    ebay_id: str
    title: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "EUR"
    condition: Optional[str] = None
    image_url: Optional[str] = None
    url: Optional[str] = None
    seller_name: Optional[str] = None


@wishlist_router.get("")
def list_wishlist(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return all wishlist items for the current user."""
    items = (
        db.query(WishlistItem)
        .filter(WishlistItem.user_id == user.id)
        .order_by(WishlistItem.added_at.desc())
        .all()
    )
    return {
        "count": len(items),
        "items": [
            {
                "id": item.id,
                "ebay_id": item.ebay_id,
                "title": item.title,
                "price": item.price,
                "currency": item.currency,
                "condition": item.condition,
                "image_url": item.image_url,
                "url": item.url,
                "seller_name": item.seller_name,
                "added_at": item.added_at.isoformat() if item.added_at else None,
            }
            for item in items
        ],
    }


@wishlist_router.post("")
def add_to_wishlist(
    body: AddItemRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a product to the current user's wishlist."""
    existing = (
        db.query(WishlistItem)
        .filter(WishlistItem.user_id == user.id, WishlistItem.ebay_id == body.ebay_id)
        .first()
    )
    if existing:
        return {"status": "already_exists", "id": existing.id}

    item = WishlistItem(
        user_id=user.id,
        ebay_id=body.ebay_id,
        title=body.title,
        price=body.price,
        currency=body.currency or "EUR",
        condition=body.condition,
        image_url=body.image_url,
        url=body.url,
        seller_name=body.seller_name,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    logger.info("Wishlist item added via REST: user=%s, ebay_id=%s", user.id, body.ebay_id)
    return {"status": "added", "id": item.id}


@wishlist_router.delete("/{ebay_id}")
def remove_from_wishlist(
    ebay_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a product from the current user's wishlist by its eBay ID."""
    item = (
        db.query(WishlistItem)
        .filter(WishlistItem.user_id == user.id, WishlistItem.ebay_id == ebay_id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Prodotto non trovato nella wishlist.")
    db.delete(item)
    db.commit()
    logger.info("Wishlist item removed via REST: user=%s, ebay_id=%s", user.id, ebay_id)
    return {"status": "removed"}
