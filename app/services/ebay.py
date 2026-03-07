from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
EBAY_ENV = os.getenv("EBAY_ENV", "sandbox").strip().lower()
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_IT").strip()

REQUEST_TIMEOUT = int(os.getenv("EBAY_REQUEST_TIMEOUT", "15"))
MAX_PAGE_SIZE = min(int(os.getenv("EBAY_PAGE_SIZE", "20")), 200)
MAX_OFFSET_PAGES = int(os.getenv("EBAY_MAX_OFFSET_PAGES", "3"))

APPROX_PRICE_PCT = float(os.getenv("APPROX_PRICE_PCT", "0.2"))
APPROX_PRICE_MIN_DELTA = float(os.getenv("APPROX_PRICE_MIN_DELTA", "10"))

if EBAY_ENV == "production":
    OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
else:
    OAUTH_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"


# ============================================================
# HTTP SESSION
# ============================================================

_SESSION = requests.Session()


# ============================================================
# TOKEN CACHE
# ============================================================

_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
}


# ============================================================
# UTILS
# ============================================================

def _normalize_numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_keep_order(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            continue
        if ebay_id in seen:
            continue
        seen.add(ebay_id)
        out.append(item)

    return out


# ============================================================
# OAUTH
# ============================================================

def _get_oauth_token(force_refresh: bool = False) -> str:
    global _token_cache

    now = time.time()

    if (
        not force_refresh
        and _token_cache["access_token"]
        and now < float(_token_cache["expires_at"])
    ):
        return _token_cache["access_token"]

    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        raise RuntimeError("EBAY_CLIENT_ID o EBAY_CLIENT_SECRET mancanti")

    auth_string = f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_auth}",
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope",
    }

    response = _SESSION.post(
        OAUTH_URL,
        headers=headers,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(f"OAuth error {response.status_code}: {response.text}")

    token_data = response.json()
    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in", 7200)

    if not access_token:
        raise RuntimeError("OAuth response without access_token")

    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = now + float(expires_in) - 60

    return access_token


# ============================================================
# FILTER BUILDERS
# ============================================================

def _expand_approx(value: float) -> Tuple[float, float]:
    delta = max(value * APPROX_PRICE_PCT, APPROX_PRICE_MIN_DELTA)
    return round(value - delta, 2), round(value + delta, 2)


def _build_price_filter(constraints: List[Dict[str, Any]]) -> Optional[str]:
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    for c in constraints:
        if c.get("type") != "price":
            continue

        op = c.get("operator")
        val = c.get("value")

        if op == "<=":
            max_price = _normalize_numeric(val)

        elif op == ">=":
            min_price = _normalize_numeric(val)

        elif op == "between" and isinstance(val, list) and len(val) == 2:
            left = _normalize_numeric(val[0])
            right = _normalize_numeric(val[1])

            if left is not None and right is not None:
                min_price = min(left, right)
                max_price = max(left, right)

    if min_price is not None and min_price <= 1:
        min_price = None

    if min_price is None and max_price is None:
        return None

    if min_price is None:
        return f"price:[..{max_price}]"

    if max_price is None:
        return f"price:[{min_price}..]"

    return f"price:[{min_price}..{max_price}]"


def _build_condition_filter(constraints: List[Dict[str, Any]]) -> Optional[str]:
    mapping = {
        "new": "1000",
        "refurbished": "2000",
        "used": "3000",
    }

    for c in constraints:
        if c.get("type") != "condition":
            continue

        val = str(c.get("value", "")).lower().strip()
        if val in mapping:
            return f"conditionIds:{{{mapping[val]}}}"

    return None


def _build_filter_string(constraints: List[Dict[str, Any]]) -> Optional[str]:
    filters: List[str] = []

    price_filter = _build_price_filter(constraints)
    if price_filter:
        filters.append(price_filter)
        filters.append("priceCurrency:EUR")

    condition_filter = _build_condition_filter(constraints)
    if condition_filter:
        filters.append(condition_filter)

    if not filters:
        return None

    return ",".join(filters)


# ============================================================
# QUERY BUILDING
# ============================================================

def _build_query(parsed: Dict[str, Any]) -> str:
    parts: List[str] = []

    brands = parsed.get("brands") or []
    product = parsed.get("product")
    semantic_query = parsed.get("semantic_query")
    original_query = parsed.get("original_query")

    if brands:
        parts.extend(str(b).strip() for b in brands if str(b).strip())

    if product:
        parts.append(str(product).strip())

    if not parts and semantic_query:
        parts.append(str(semantic_query).strip())

    if not parts and original_query:
        parts.append(str(original_query).strip())

    query = " ".join(p for p in parts if p).strip()
    return query


def _build_sort(preferences: List[Dict[str, Any]]) -> Optional[str]:
    """
    Hook semplice per preferenze future.
    Per ora lasciamo best match implicito.
    """
    for pref in preferences:
        if pref.get("type") == "price":
            # potresti scegliere "price" se vuoi
            return None
    return None


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    price_info = item.get("price") or {}
    seller_info = item.get("seller") or {}
    image_info = item.get("image") or {}

    return {
        "ebay_id": item.get("itemId"),
        "title": _clean_text(item.get("title")),
        "price": _normalize_numeric(price_info.get("value")) or 0,
        "currency": _clean_text(price_info.get("currency")),
        "condition": _clean_text(item.get("condition")),
        "seller_name": _clean_text(seller_info.get("username")),
        "seller_rating": _normalize_numeric(seller_info.get("feedbackPercentage")),
        "url": _clean_text(item.get("itemWebUrl")),
        "image_url": _clean_text(image_info.get("imageUrl")),
        "brand": _clean_text(item.get("brand")),
    }


# ============================================================
# RAW SEARCH
# ============================================================

def _perform_search_request(
    token: str,
    query: str,
    filter_string: Optional[str],
    limit: int,
    offset: int,
    sort: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }

    params: Dict[str, Any] = {
        "q": query,
        "limit": limit,
        "offset": offset,
    }

    if filter_string:
        params["filter"] = filter_string

    if sort:
        params["sort"] = sort

    response = _SESSION.get(
        SEARCH_URL,
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 401:
        # token scaduto / invalido
        token = _get_oauth_token(force_refresh=True)
        headers["Authorization"] = f"Bearer {token}"

        response = _SESSION.get(
            SEARCH_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    if response.status_code != 200:
        raise RuntimeError(f"eBay search error {response.status_code}: {response.text}")

    return response.json()


# ============================================================
# PUBLIC SEARCH API
# ============================================================

def search_items(
    parsed_query: Dict[str, Any],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search eBay items with optimized pagination:
    - reuse oauth token
    - reuse HTTP session
    - stop early when enough items collected
    - avoid deep pagination by default
    """
    query = _build_query(parsed_query)
    if not query:
        return []

    token = _get_oauth_token()

    constraints = parsed_query.get("constraints") or []
    preferences = parsed_query.get("preferences") or []

    filter_string = _build_filter_string(constraints)
    sort = _build_sort(preferences)

    wanted = max(1, int(limit))
    page_size = min(MAX_PAGE_SIZE, wanted)

    items: List[Dict[str, Any]] = []
    offset = 0
    pages_done = 0

    while len(items) < wanted and pages_done < MAX_OFFSET_PAGES:
        data = _perform_search_request(
            token=token,
            query=query,
            filter_string=filter_string,
            limit=page_size,
            offset=offset,
            sort=sort,
        )

        page_items = data.get("itemSummaries", []) or []
        if not page_items:
            break

        normalized_page = [_normalize_item(i) for i in page_items]
        items.extend(normalized_page)

        # early stop se la pagina torna meno elementi del richiesto
        if len(page_items) < page_size:
            break

        offset += page_size
        pages_done += 1

    items = _dedupe_keep_order(items)

    # trim finale
    return items[:wanted]