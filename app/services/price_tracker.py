"""
price_tracker.py
================
Background service that periodically checks the current eBay price for all
items saved in the wishlist and updates `previous_price` / `price` when a
price drop is detected.

The loop runs every PRICE_CHECK_INTERVAL_HOURS hours (default 1 h) and is
started once at application startup via asyncio.create_task() in main.py.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PRICE_CHECK_INTERVAL_HOURS: float = 1.0   # configurable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_price(raw: Optional[dict]) -> Optional[float]:
    """Extract a float price from an eBay price dict like {'value': '99.99', 'currency': 'EUR'}."""
    if not isinstance(raw, dict):
        return None
    try:
        return float(raw.get("value", 0) or 0)
    except (TypeError, ValueError):
        return None


async def _fetch_current_price(ebay_id: str) -> Optional[float]:
    """Call the eBay API to get the current price of an item. Returns None on failure."""
    try:
        from app.services.ebay import get_item_details_by_id  # type: ignore

        data: dict = await get_item_details_by_id(ebay_id)
        if not data:
            return None

        # Try the standard price fields returned by the eBay Browse API item detail
        price = (
            _parse_price(data.get("price"))
            or _parse_price(data.get("currentBidPrice"))
            or _parse_price((data.get("estimatedAvailabilities") or [{}])[0].get("price"))
        )
        return price
    except Exception as exc:
        logger.debug("price_tracker: could not fetch price for %s — %s", ebay_id, exc)
        return None


def _run_price_check_sync() -> None:
    """
    Synchronous DB pass: for each wishlist item, fetch the current eBay price.
    Uses its own DB session so it can safely run in an asyncio executor.
    """
    from app.db.database import SessionLocal  # type: ignore
    from app.models.wishlist import WishlistItem  # type: ignore

    db = SessionLocal()
    try:
        items = db.query(WishlistItem).all()
        logger.info("price_tracker: checking %d wishlist items", len(items))

        for item in items:
            if not item.ebay_id:
                continue

            # We run each fetch synchronously inside the thread — acceptable for background work
            loop = asyncio.new_event_loop()
            try:
                new_price = loop.run_until_complete(_fetch_current_price(item.ebay_id))
            finally:
                loop.close()

            if new_price is None:
                continue

            # Detect a price drop
            if item.price is not None and new_price < item.price:
                logger.info(
                    "price_tracker: PRICE DROP detected for %s: %.2f → %.2f",
                    item.ebay_id, item.price, new_price,
                )
                item.previous_price = item.price   # archive old price
                item.price = new_price             # update to new lower price
            elif new_price != item.price:
                # Price went up or item was just added — update silently
                item.price = new_price
                item.previous_price = None         # reset badge when price goes back up

            item.last_checked_at = datetime.now(timezone.utc)

        db.commit()
        logger.info("price_tracker: check cycle complete")

    except Exception as exc:
        logger.error("price_tracker: error during check cycle — %s", exc)
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

async def price_tracking_loop() -> None:
    """
    Infinite async loop launched at startup.
    Waits PRICE_CHECK_INTERVAL_HOURS between cycles to avoid hammering the API.
    """
    logger.info(
        "price_tracker: background loop started (interval=%.1fh)", PRICE_CHECK_INTERVAL_HOURS
    )
    while True:
        try:
            # Run the sync DB work in a thread so we don't block the event loop
            await asyncio.to_thread(_run_price_check_sync)
        except Exception as exc:
            logger.error("price_tracker: unhandled error in loop — %s", exc)

        await asyncio.sleep(PRICE_CHECK_INTERVAL_HOURS * 3600)
