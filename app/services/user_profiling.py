from typing import Dict
from sqlalchemy.orm import Session
from app.models.user import User


def update_user_profile(user: User, parsed: Dict, db: Session):

    if not user:
        return

    updated = False

    # -------------------------
    # BRAND PREFERENCE
    # -------------------------

    brands = parsed.get("brands") or []

    if brands:

        brand = brands[0]

        if user.favorite_brands != brand:
            user.favorite_brands = brand
            updated = True

    # -------------------------
    # PRICE PREFERENCE
    # -------------------------

    constraints = parsed.get("constraints") or []

    for c in constraints:

        if c.get("type") == "price":

            value = c.get("value")

            if value:

                try:

                    price = int(value)

                    if not user.price_preference:
                        user.price_preference = str(price)

                    else:

                        old = int(user.price_preference)

                        # media mobile
                        new_price = int((old + price) / 2)

                        user.price_preference = str(new_price)

                    updated = True

                except:
                    pass

    if updated:
        db.commit()