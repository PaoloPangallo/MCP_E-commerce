# app/services/ebay.py

import base64
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
EBAY_ENV = os.getenv("EBAY_ENV", "sandbox").strip().lower()
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_IT").strip()

REQUEST_TIMEOUT = int(os.getenv("EBAY_REQUEST_TIMEOUT", "20"))

APPROX_PRICE_PCT = float(os.getenv("APPROX_PRICE_PCT", "0.2"))
APPROX_PRICE_MIN_DELTA = float(os.getenv("APPROX_PRICE_MIN_DELTA", "10"))

if EBAY_ENV == "production":
    OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
else:
    OAUTH_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"

# ============================================================
# TOKEN CACHE
# ============================================================

_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
}


# ============================================================
# OAUTH
# ============================================================

def _get_oauth_token() -> str:

    global _token_cache

    now = time.time()

    if _token_cache["access_token"] and now < _token_cache["expires_at"]:
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

    response = requests.post(
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
# PRICE FILTER
# ============================================================

def _normalize_numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except:
        return None


def _expand_approx(value: float) -> Tuple[float, float]:

    delta = max(value * APPROX_PRICE_PCT, APPROX_PRICE_MIN_DELTA)
    return round(value - delta, 2), round(value + delta, 2)


def _build_price_filter(constraints: List[Dict[str, Any]]) -> Optional[str]:

    min_price = None
    max_price = None

    for c in constraints:

        if c.get("type") != "price":
            continue

        op = c.get("operator")
        val = c.get("value")

        if op == "<=":
            max_price = _normalize_numeric(val)

        elif op == ">=":
            min_price = _normalize_numeric(val)

        elif op == "between" and isinstance(val, list):

            if len(val) == 2:
                left = _normalize_numeric(val[0])
                right = _normalize_numeric(val[1])

                if left and right:
                    min_price = min(left, right)
                    max_price = max(left, right)

    # Sanity check: if prices are near 0, treat as None (no limit)
    if min_price is not None and min_price <= 0.1:
        min_price = None
    if max_price is not None and max_price <= 0.1:
        max_price = None

    if min_price is None and max_price is None:
        return None

    if min_price is None:
        return f"price:[..{max_price}]"
    if max_price is None:
        return f"price:[{min_price}..]"

    return f"price:[{min_price}..{max_price}]"


# ============================================================
# CONDITION FILTER
# ============================================================

def _build_condition_filter(constraints: List[Dict[str, Any]]) -> Optional[str]:

    mapping = {
        "new": "1000",
        "refurbished": "2000",
        "used": "3000",
    }

    for c in constraints:

        if c.get("type") != "condition":
            continue

        val = str(c.get("value", "")).lower()

        if val in mapping:
            return f"conditionIds:{{{mapping[val]}}}"

    return None


# ============================================================
# BUILD EBAY FILTER STRING
# ============================================================

def _build_filter_string(constraints: List[Dict[str, Any]]) -> Optional[str]:

    filters = []

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
# BUILD EBAY QUERY
# ============================================================

def _build_query(parsed: Dict[str, Any]) -> str:
    parts = []
    
    # Extract Data from parsed
    brands = parsed.get("brands") or []
    product = parsed.get("product", "")
    semantic = parsed.get("semantic_query", "")
    sem_parts = str(semantic).split()
    
    # Add values from compatibilities (e.g., Size: 42, Color: Black, TargetModel: iPhone)
    compatibilities = parsed.get("compatibilities") or {}
    comp_values = [str(v) for v in compatibilities.values() if v]
    
    # Merge Logic: Add parts from brands + product + compatibilities, then add semantic parts
    raw_candidates = []
    if brands:
        if isinstance(brands, list): raw_candidates.extend([str(b) for b in brands])
        else: raw_candidates.append(str(brands))
    if product:
        if isinstance(product, list): raw_candidates.extend([str(p) for p in product])
        else: raw_candidates.append(str(product))
    if comp_values:
        raw_candidates.extend(comp_values)
    
    # Pre-add semantic query parts
    raw_candidates.extend(str(semantic).split())

    # Stopwords list: generic words that pollute eBay search results
    stopwords = {
        "modello", "tipo", "articolo", "cercare", "cerco", "vorrei", "trovami", "oggetto", 
        "chiesto", "richiesta", "precedente", "quello", "quelli", "stessi", "altri", "altre",
        "ciao", "grazie", "per", "con", "dei", "delle", "degli", "un", "una", "uno", 
        "li", "lo", "la", "le", "il", "i", "gli", "da", "di", "a", "in", "su", "nel", "nella", 
        "chi", "cosa", "come", "dove", "quando", "perché", "perche", "puoi", "fare", "fai", "fatto",
        "aiutarmi", "aiuto", "assistenza", "cercami", "trova", "voglio", "serve", "servirebbe",
        "mille", "benvenuto", "gentilissimo", "ok", "va", "bene", "figo", "perfetto", "ottimo",
        "cerchiamo", "vediamo", "prodotto", "help", "search", "shopping", "assistant", "ebay",
        "originale", "keywords", "taglia", "numero", "size", "percaso", "caso", "colore", "color", 
        "ce", "disponibile", "disponibili", "trovi", "trovare", "esiste", "esistono", "sono", "sia", "fosse"
    }

    seen_norms = set()
    cleaned_parts = []
    
    for candidate in raw_candidates:
        # Split candidate in case it contains multiple words (e.g. from brands list)
        for word in str(candidate).split():
            # NORMALIZE: remove punctuation for check (e.g. "ciao," -> "ciao")
            w_clean = re.sub(r"[^\w]", "", word.lower())
            
            if not w_clean or len(w_clean) < 2:
                continue
                
            if w_clean in stopwords:
                continue
                
            if w_clean not in seen_norms:
                cleaned_parts.append(word)
                seen_norms.add(w_clean)

    # Category-Specific Standardization (Safety Layer)
    query_str = " ".join(cleaned_parts)
    product_low = str(product).lower()
    
    # 1. Shoes: "taglia 44" -> "EU 44", Gender detection
    if any(x in product_low for x in ["scarpe", "sneakers", "stivali", "calzature"]):
        # Match standalone numbers 35-48 or "taglia/numero 43"
        query_str = re.sub(r"\b(?:taglia|numero|taglie)\s+(\d{2})\b", r"EU \1", query_str, flags=re.IGNORECASE)
        # Gender standardization for shoes
        if "uomo" in query_str.lower() and "men's" not in query_str.lower():
            query_str += " Men's"
        elif "donna" in query_str.lower() and "women's" not in query_str.lower():
            query_str += " Women's"
            
    # 2. Jeans/Pants: "taglia X lunghezza Y" -> "WX LY"
    elif any(x in product_low for x in ["jeans", "pantaloni", "pantalone"]):
        t_match = re.search(r"taglia\s+(\d+)", query_str, re.IGNORECASE)
        l_match = re.search(r"lunghezza\s+(\d+)", query_str, re.IGNORECASE)
        if t_match and l_match:
            query_str = re.sub(r"taglia\s+\d+", f"W{t_match.group(1)}", query_str, flags=re.IGNORECASE)
            query_str = re.sub(r"lunghezza\s+\d+", f"L{l_match.group(1)}", query_str, flags=re.IGNORECASE)
        elif t_match:
            query_str = re.sub(r"taglia\s+\d+", f"W{t_match.group(1)}", query_str, flags=re.IGNORECASE)
    
    # Final cleanup: deduplicate again after regex replacements
    final_words = []
    seen = set()
    for w in query_str.split():
        w_norm = re.sub(r"[^\w]", "", w.lower())
        if w_norm and w_norm not in seen:
            final_words.append(w)
            seen.add(w_norm)
            
    final_q = " ".join(final_words).strip()
    
    if not final_q:
        final_q = parsed.get("original_query", "")

    return final_q


# ============================================================
# NORMALIZE RESPONSE
# ============================================================

def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:

    price_info = item.get("price") or {}
    seller_info = item.get("seller") or {}
    image_info = item.get("image") or {}

    return {
        "ebay_id": item.get("itemId"),
        "title": item.get("title"),
        "price": _normalize_numeric(price_info.get("value")) or 0,
        "currency": price_info.get("currency"),
        "condition": item.get("condition"),
        "seller_name": seller_info.get("username"),
        "seller_rating": _normalize_numeric(seller_info.get("feedbackPercentage")),
        "url": item.get("itemWebUrl"),
        "image_url": image_info.get("imageUrl"),
        "brand": item.get("brand"),
    }


# ============================================================
# PUBLIC SEARCH API
# ============================================================

def search_items(
    parsed_query: Dict[str, Any],
    limit: int = 30,
) -> List[Dict[str, Any]]:

    token = _get_oauth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }

    query = _build_query(parsed_query)
    constraints = parsed_query.get("constraints", [])
    
    items = []
    offset = 0
    page_size = 20

    while len(items) < limit:
        params = {
            "q": query,
            "limit": page_size,
            "offset": offset,
        }

        filter_string = _build_filter_string(constraints)
        if filter_string:
            params["filter"] = filter_string

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"eBay Search Request: URL={SEARCH_URL} PARAMS={params}")

        response = requests.get(
            SEARCH_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        logger.info(f"eBay Search Response Status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"eBay API error {response.status_code}: {response.text}")
            break

        data = response.json()
        page_items = data.get("itemSummaries", [])
        total = data.get("total", 0)

        logger.info(f"eBay Results: {len(page_items)} items on this page (Total available: {total})")

        if not page_items:
            break

        items.extend(page_items)
        offset += page_size

    items = items[:limit]
    return [_normalize_item(i) for i in items]