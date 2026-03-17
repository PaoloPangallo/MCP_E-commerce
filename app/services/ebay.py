import asyncio
import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
EBAY_ENV = os.getenv("EBAY_ENV", "sandbox").strip().lower()
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_IT").strip()

REQUEST_TIMEOUT = float(os.getenv("EBAY_REQUEST_TIMEOUT", "15")) # Changed to float for httpx client

MAX_PAGE_SIZE = min(int(os.getenv("EBAY_PAGE_SIZE", "20")), 200)
MAX_OFFSET_PAGES = int(os.getenv("EBAY_MAX_OFFSET_PAGES", "3"))

APPROX_PRICE_PCT = float(os.getenv("APPROX_PRICE_PCT", "0.2"))
APPROX_PRICE_MIN_DELTA = float(os.getenv("APPROX_PRICE_MIN_DELTA", "10"))

if EBAY_ENV == "production":
    OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    ITEM_URL = "https://api.ebay.com/buy/browse/v1/item/"
else:
    OAUTH_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"
    ITEM_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item/"


# ============================================================
# HTTP CLIENT (SINGLETON)
# ============================================================

_shared_client: Optional[httpx.AsyncClient] = None

async def init_http_client() -> None:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

async def close_http_client() -> None:
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None

def get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        logger.warning("EBAY shared HTTP client not initialized! Autocreating fallback.")
        _shared_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _shared_client

# ============================================================
# TOKEN CACHE
# ============================================================

_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
}
_token_lock = asyncio.Lock()


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


from app.utils.text import clean_text as _clean_text


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

async def _get_oauth_token(force_refresh: bool = False) -> str:
    global _token_cache

    # Fast path: check cache without lock
    now = time.time()
    if (
        not force_refresh
        and _token_cache["access_token"]
        and now < float(_token_cache["expires_at"])
    ):
        return _token_cache["access_token"]

    # Slow path: acquire lock for refresh
    async with _token_lock:
        # Double-check after acquiring lock (another coroutine may have refreshed)
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

        client = get_client()
        response = await client.post(
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


def _build_aspect_filter(constraints: List[Dict[str, Any]]) -> List[str]:
    """
    Builds a list of aspect filters.
    Syntax: aspect_filter:Name:{Value}
    """
    aspect_filters = []
    for c in constraints:
        if c.get("type") == "aspect":
            name = c.get("name")
            value = c.get("value")
            if name and value:
                aspect_filters.append(f"aspect_filter:{name}:{{{value}}}")
    return aspect_filters


def _build_filter_string(constraints: List[Dict[str, Any]]) -> Optional[str]:
    filters: List[str] = []

    price_filter = _build_price_filter(constraints)
    if price_filter:
        filters.append(price_filter)
        filters.append("priceCurrency:EUR")

    condition_filter = _build_condition_filter(constraints)
    if condition_filter:
        filters.append(condition_filter)

    aspect_filters = _build_aspect_filter(constraints)
    if aspect_filters:
        filters.extend(aspect_filters)

    if not filters:
        return None

    return ",".join(filters)


# ============================================================
# QUERY BUILDING
# ============================================================

def _build_query(parsed: Dict[str, Any]) -> str:
    # 1) Elaboriamo i componenti principali
    parts: List[str] = []
    
    brands = parsed.get("brands") or []
    product = parsed.get("product")
    
    if brands:
        parts.extend(str(b).strip() for b in brands if str(b).strip())
    if product:
        parts.append(str(product).strip())
        
    # 2) Includiamo constraints che non sono price/condition (spesso sono feature testuali)
    constraints = parsed.get("constraints") or []
    for c in constraints:
        ctype = c.get("type")
        val = c.get("value")
        if ctype not in ("price", "condition", "aspect") and val:
            if isinstance(val, list):
                parts.extend(str(v) for v in val)
            else:
                parts.append(str(val))
    
    # 3) Se ancora vuoto, usiamo semantic/original query
    if not parts:
        fallback = parsed.get("semantic_query") or parsed.get("original_query")
        if fallback:
            parts.append(str(fallback).strip())
            
    # 4) DEDUPLICAZIONE PAROLE MANTENENDO ORDINE
    # Esempio: "iPhone iPhone 13 128gb" -> "iPhone 13 128gb"
    final_tokens: List[str] = []
    seen_tokens_low = set()
    
    raw_query = " ".join(parts)
    for word in raw_query.split():
        w_low = word.lower()
        if w_low not in seen_tokens_low:
            final_tokens.append(word)
            seen_tokens_low.add(w_low)
            
    return " ".join(final_tokens).strip()


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

async def _perform_search_request(
    client: httpx.AsyncClient,
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
        "fieldgroups": "FULL",
    }

    if filter_string:
        params["filter"] = filter_string
    if sort:
        params["sort"] = sort

    try:
        response = await client.get(
            SEARCH_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.TimeoutException:
        logger.error("EBAY TIMEOUT")
        raise RuntimeError("EBAY timeout")
    except Exception as exc:
        logger.exception("EBAY CONNECTION ERROR")
        raise RuntimeError("EBAY connection error")

    if response.status_code == 401:
        token = await _get_oauth_token(force_refresh=True)
        headers["Authorization"] = f"Bearer {token}"
        response = await client.get(
            SEARCH_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    if response.status_code != 200:
        logger.error("EBAY ERROR %s %s", response.status_code, response.text)
        raise RuntimeError(f"eBay search error {response.status_code}")

    return response.json()


# ============================================================
# PUBLIC SEARCH API
# ============================================================

async def search_items(
    parsed_query: Dict[str, Any],
    limit: int = 20,
) -> Dict[str, Any]:

    logger.info("EBAY SEARCH START")
    query = _build_query(parsed_query)
    if not query:
        return {"itemSummaries": [], "aspectDistributions": []}

    try:
        token = await _get_oauth_token()
    except Exception:
        return {"itemSummaries": [], "aspectDistributions": []}

    constraints = parsed_query.get("constraints") or []
    preferences = parsed_query.get("preferences") or []
    filter_string = _build_filter_string(constraints)
    sort = _build_sort(preferences)

    wanted = max(1, int(limit))
    page_size = min(MAX_PAGE_SIZE, wanted)
    items: List[Dict[str, Any]] = []
    offset = 0
    pages_done = 0
    aspects: List[Dict[str, Any]] = []

    client = get_client()
    while len(items) < wanted and pages_done < MAX_OFFSET_PAGES:
        try:
            data = await _perform_search_request(
                client=client,
                token=token,
                query=query,
                filter_string=filter_string,
                limit=page_size,
                offset=offset,
                sort=sort,
            )
        except Exception:
            break

        page_items = data.get("itemSummaries", []) or []
        logger.info("eBay page response keys=%s, page_items=%d", list(data.keys()), len(page_items))
        if pages_done == 0:
            refinement = data.get("refinement") or {}
            aspects = refinement.get("aspectDistributions", []) or []
        
        if not page_items:
            break

        items.extend([_normalize_item(i) for i in page_items])
        if len(page_items) < page_size:
            break

        offset += page_size
        pages_done += 1

    items = _dedupe_keep_order(items)
    return {
        "itemSummaries": items[:wanted],
        "aspectDistributions": aspects,
    }


# ============================================================
# ITEM DETAILS API
# ============================================================

async def get_item_details(item_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches full details of a specific item from eBay using the getItem endpoint.
    """
    logger.info("EBAY GET ITEM START | item_id=%s", item_id)

    try:
        token = await _get_oauth_token()
    except Exception:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }

    encoded_item_id = urllib.parse.quote(item_id)
    url = f"{ITEM_URL}{encoded_item_id}"

    client = get_client()
    try:
        response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 401:
            token = await _get_oauth_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
        if response.status_code != 200:
            logger.warning("EBAY GET ITEM ERROR | status=%s | body=%s", response.status_code, response.text)
            return None

        data = response.json()
        item_details = {
            "item_id": data.get("itemId"),
            "title": data.get("title"),
            "description": data.get("description"),
            "item_specifics": data.get("localizedAspects", []),
            "seller": data.get("seller", {}),
            "price": data.get("price", {}),
            "brand": data.get("brand"),
            "image": data.get("image", {}),
            "additional_images": data.get("additionalImages", []),
            "item_url": data.get("itemWebUrl"),
            "condition": data.get("condition"),
        }
        
        # Enrich with similar items
        try:
            item_details["similar_items"] = await get_similar_items(item_id)
        except Exception as e:
            logger.warning("Failed to fetch similar items for %s: %s", item_id, e)
            item_details["similar_items"] = []
            
        return item_details
    except Exception as e:
        logger.error("EBAY GET ITEM EXCEPTION | item_id=%s | error=%s", item_id, e)
        return None

# ============================================================
# SIMILAR ITEMS API
# ============================================================

async def get_similar_items(item_id: str) -> List[Dict[str, Any]]:
    """
    Fetches items similar to the given item_id using eBay's Browse API.
    """
    logger.info("EBAY GET SIMILAR ITEMS START | item_id=%s", item_id)

    try:
        token = await _get_oauth_token()
    except Exception:
        return []

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }

    # eBay Browse API endpoint for similar items
    # Note: Using product_id or seeds might be better in some cases, 
    # but for simplicity we stick to item_id if the API supports it or use a search fallback.
    # Actually, Browse API provides 'get_similar_items' via 'item' resource.
    encoded_item_id = urllib.parse.quote(item_id)
    url = f"{ITEM_URL}{encoded_item_id}/get_similar_items"

    client = get_client()
    try:
        response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 401:
            token = await _get_oauth_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            logger.warning("EBAY GET SIMILAR ITEMS ERROR | status=%s | body=%s", response.status_code, response.text)
            return []

        data = response.json()
        raw_items = data.get("itemSummaries", []) or []
            
        return [_normalize_item(i) for i in raw_items]
    except Exception as e:
        logger.error("EBAY GET SIMILAR ITEMS EXCEPTION | %s", e)
        return []




# ============================================================
# SHIPPING COSTS API
# ============================================================

async def get_shipping_costs(item_id: str, country_code: str, zip_code: str) -> Optional[Dict[str, Any]]:
    """
    Fetches precise shipping costs using an async client.
    """
    try:
        token = await _get_oauth_token()
    except Exception:
        return None

    user_ctx = f"contextualLocation=country={country_code},zip={zip_code}"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
        "X-EBAY-C-ENDUSERCTX": user_ctx,
    }

    encoded_item_id = urllib.parse.quote(item_id)
    url = f"{ITEM_URL}{encoded_item_id}"
    client = get_client()
    try:
        response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 401:
            token = await _get_oauth_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            return None

        data = response.json()
        return {
            "item_id": item_id,
            "shipping_options": data.get("shippingOptions", []),
            "item_location": data.get("itemLocation", {}),
        }
    except Exception:
        return None