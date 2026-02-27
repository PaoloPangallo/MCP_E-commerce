# app/services/ebay.py

import base64
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
EBAY_ENV = os.getenv("EBAY_ENV", "sandbox").strip().lower()
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_IT").strip()

REQUEST_TIMEOUT = 20

if EBAY_ENV == "production":
    OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
else:
    OAUTH_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"

# ============================================================
# TOKEN CACHE (semplice, in-memory)
# ============================================================

_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
}


# ============================================================
# OAUTH
# ============================================================

def _get_oauth_token() -> str:
    """
    Restituisce un access token valido.
    Usa una cache in-memory finché il token non scade.
    """
    global _token_cache

    now = time.time()

    if _token_cache["access_token"] and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        raise RuntimeError("EBAY_CLIENT_ID o EBAY_CLIENT_SECRET mancanti nel .env")

    auth_string = f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_auth}",
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope",
    }

    response = requests.post(
        OAUTH_URL,
        headers=headers,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(f"OAuth error: {response.status_code} {response.text}")

    token_data = response.json()

    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in", 7200)

    if not access_token:
        raise RuntimeError(f"OAuth response without access_token: {token_data}")

    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = now + float(expires_in) - 60  # buffer sicurezza

    return access_token


# ============================================================
# CONSTRAINTS -> EBAY FILTERS
# ============================================================

def _normalize_numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_price_filter(constraints: List[Dict[str, Any]]) -> Optional[str]:
    """
    Converte constraints di prezzo nel formato eBay Browse API.

    Esempi:
    - price:[..600]
    - price:[300..]
    - price:[300..600]
    """
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    for c in constraints:
        if c.get("type") != "price":
            continue

        op = c.get("operator")
        val = c.get("value")

        if op == "<=":
            candidate = _normalize_numeric(val)
            if candidate is not None:
                max_price = candidate

        elif op == ">=":
            candidate = _normalize_numeric(val)
            if candidate is not None:
                min_price = candidate

        elif op == "between" and isinstance(val, list) and len(val) == 2:
            left = _normalize_numeric(val[0])
            right = _normalize_numeric(val[1])
            if left is not None and right is not None:
                min_price = min(left, right)
                max_price = max(left, right)

    # Ignora il classico artefatto inutile ">= 0"
    if min_price == 0:
        min_price = None

    if min_price is None and max_price is None:
        return None

    if min_price is None:
        return f"price:[..{max_price}]"

    if max_price is None:
        return f"price:[{min_price}..]"

    return f"price:[{min_price}..{max_price}]"


def _build_condition_filter(constraints: List[Dict[str, Any]]) -> Optional[str]:
    """
    Converte condition -> conditionIds eBay Browse API.
    """
    mapping = {
        "new": "1000",
        "refurbished": "2000",
        "used": "3000",
    }

    for c in constraints:
        if c.get("type") != "condition":
            continue

        val = str(c.get("value", "")).strip().lower()
        condition_id = mapping.get(val)
        if condition_id:
            return f"conditions:{{{condition_id}}}"

    return None


def _build_filter_string(constraints: List[Dict[str, Any]]) -> Optional[str]:
    filters: List[str] = []

    price_filter = _build_price_filter(constraints)
    if price_filter:
        filters.append(price_filter)

    condition_filter = _build_condition_filter(constraints)
    if condition_filter:
        filters.append(condition_filter)

    if not filters:
        return None

    return ",".join(filters)


# ============================================================
# RESPONSE NORMALIZATION
# ============================================================

def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    price_info = item.get("price") or {}
    seller_info = item.get("seller") or {}
    image_info = item.get("image") or {}

    price_value = _normalize_numeric(price_info.get("value")) or 0.0
    seller_rating = _normalize_numeric(seller_info.get("feedbackPercentage"))

    return {
        "ebay_id": item.get("itemId"),
        "title": item.get("title"),
        "price": price_value,
        "currency": price_info.get("currency"),
        "condition": item.get("condition"),
        "seller_name": seller_info.get("username"),
        "seller_rating": seller_rating,
        "url": item.get("itemWebUrl"),
        "image_url": image_info.get("imageUrl"),
    }


# ============================================================
# PUBLIC API
# ============================================================

def search_items(
    query_text: str,
    constraints: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:

    if not query_text or not query_text.strip():
        raise ValueError("query_text non può essere vuota")

    if limit <= 0:
        limit = 10

    token = _get_oauth_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }

    params: Dict[str, Any] = {
        "q": query_text.strip(),
        "limit": limit,
    }

    filter_string = _build_filter_string(constraints)
    if filter_string:
        params["filter"] = filter_string

    response = requests.get(
        SEARCH_URL,
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    # ============================================================
    # GESTIONE ERRORI INTELLIGENTE
    # ============================================================

    # 401 → token probabilmente scaduto → retry una volta
    if response.status_code == 401:
        _token_cache["access_token"] = None
        token = _get_oauth_token()

        headers["Authorization"] = f"Bearer {token}"

        response = requests.get(
            SEARCH_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    # Rate limit
    if response.status_code == 429:
        raise RuntimeError("eBay rate limit exceeded (429)")

    # Server error
    if 500 <= response.status_code < 600:
        raise RuntimeError(f"eBay server error {response.status_code}")

    if response.status_code != 200:
        raise RuntimeError(f"Search error: {response.status_code} {response.text}")

    data = response.json()
    items = data.get("itemSummaries", [])

    return [_normalize_item(item) for item in items]