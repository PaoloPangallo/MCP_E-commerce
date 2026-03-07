from __future__ import annotations

from typing import Dict, Optional

from sqlalchemy.orm import Session

from app.models.user import User


def _extract_price_signal(parsed: Dict) -> Optional[int]:
    constraints = parsed.get("constraints") or []

    for c in constraints:
        if c.get("type") != "price":
            continue

        value = c.get("value")

        try:
            if isinstance(value, (int, float)):
                return int(value)

            if isinstance(value, list) and len(value) == 2:
                nums = [int(float(v)) for v in value]
                return int(sum(nums) / 2)
        except Exception:
            continue

    return None


def update_user_profile(user: User, parsed: Dict, db: Session) -> bool:
    """
    Update user profile fields in-memory only.
    NO commit here. Commit is delegated to the outer pipeline.
    Returns True if something changed.
    """
    if not user:
        return False

    updated = False

    # --------------------------------------------------
    # BRAND PREFERENCE
    # --------------------------------------------------

    brands = parsed.get("brands") or []
    if brands:
        new_brand = str(brands[0]).strip()

        if new_brand and user.favorite_brands != new_brand:
            user.favorite_brands = new_brand
            updated = True

    # --------------------------------------------------
    # PRICE PREFERENCE
    # --------------------------------------------------

    price_signal = _extract_price_signal(parsed)
    if price_signal:
        try:
            if not user.price_preference:
                user.price_preference = str(price_signal)
                updated = True
            else:
                old = int(user.price_preference)
                new_price = int((old + price_signal) / 2)

                if str(new_price) != str(user.price_preference):
                    user.price_preference = str(new_price)
                    updated = True
        except Exception:
            pass

    if updated:
        db.add(user)

    return updated