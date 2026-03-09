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
    TAXONOMY_URL = "https://api.ebay.com/commerce/taxonomy/v1"
else:
    OAUTH_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"
    TAXONOMY_URL = "https://api.sandbox.ebay.com/commerce/taxonomy/v1"

# ID albero categorie predefinito (101 per l'Italia)
EBAY_IT_TREE_ID = "101"

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

def _build_query(parsed: Dict[str, Any], light_mode: bool = False) -> str:
    """Costruisce una query ottimizzata per eBay rimuovendo rumore e ridondanze."""
    
    # In light mode usiamo solo le basi per massimizzare i risultati (utile in caso di 0 found)
    if light_mode:
        brands = parsed.get("brands") or []
        product = parsed.get("product", "")
        base = " ".join(brands) + " " + str(product)
        return base.strip()

    semantic = parsed.get("semantic_query", "").strip()
    
    # Se abbiamo già una semantic_query di qualità prodotta dal parser LLM, la usiamo come base
    raw_text = semantic
    if not raw_text:
        brands = parsed.get("brands") or []
        product = parsed.get("product", "")
        compatibilities = parsed.get("compatibilities") or {}
        comp_values = " ".join([str(v) for v in compatibilities.values() if v])
        raw_text = f"{' '.join(brands)} {product} {comp_values}"

    # Stopwords list: parole che sporcano la ricerca
    stopwords = {
        "modello", "tipo", "articolo", "cercare", "cerco", "vorrei", "trovami", "oggetto", 
        "ciao", "grazie", "per", "con", "un", "una", "uno", "il", "i", "gli", "da", "di", "a", 
        "shopping", "ebay", "prodotto", "originale", "taglia", "numero", "size", "colore", "color",
        "men", "mens", "women", "womens"
    }

    seen_norms = set()
    cleaned_parts = []
    
    for word in raw_text.split():
        # Preserviamo il '-' se è all'inizio per keyword negative di eBay
        is_negative = word.startswith("-")
        clean_word = word[1:] if is_negative else word
        
        w_norm = re.sub(r"[^\w]", "", clean_word.lower())
        if not w_norm or w_norm in stopwords:
            continue
            
        # Allow single chars only if they are sizes (s, m, l, x) or digits
        if len(w_norm) < 2:
            if w_norm not in {"s", "m", "l", "x"} and not w_norm.isdigit():
                continue
            
        final_word = f"-{w_norm}" if is_negative else word
        if final_word.lower() not in seen_norms:
            cleaned_parts.append(final_word)
            seen_norms.add(final_word.lower())

    q_str = " ".join(cleaned_parts)
    # Preferiamo l'italiano per eBay IT
    if "uomo" in q_str.lower():
        q_str = q_str.replace("Uomo", "").replace("uomo", "").strip() + " uomo"
    elif "donna" in q_str.lower():
        q_str = q_str.replace("Donna", "").replace("donna", "").strip() + " donna"

    return q_str.strip() or parsed.get("original_query", "")


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
    category_id: Optional[str] = None,
    ignore_constraints: bool = False
) -> List[Dict[str, Any]]:

    token = _get_oauth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }

    # Primo tentativo: Full Query
    query = _build_query(parsed_query, light_mode=False)
    items = _execute_ebay_search(query, parsed_query, limit, headers, category_id=category_id, ignore_constraints=ignore_constraints)
    
    # Fallback retry se 0 risultati
    if not items:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("NO RESULTS FOUND. Retrying with LIGHT QUERY...")
        query_light = _build_query(parsed_query, light_mode=True)
        if query_light != query:
            items = _execute_ebay_search(query_light, parsed_query, limit, headers, category_id=category_id, ignore_constraints=ignore_constraints)

    items = items[:limit]
    return [_normalize_item(i) for i in items]

def _execute_ebay_search(query: str, parsed_query: Dict[str, Any], limit: int, headers: Dict[str, str], category_id: Optional[str] = None, ignore_constraints: bool = False) -> List[Dict[str, Any]]:
    """Logica interna di loop per la ricerca paginata."""
    items = []
    offset = 0
    page_size = 20
    constraints = parsed_query.get("constraints", []) if not ignore_constraints else []
    
    while len(items) < limit:
        params = {
            "q": query,
            "limit": page_size,
            "offset": offset,
        }
        
        if category_id:
            params["category_ids"] = category_id

        filter_string = _build_filter_string(constraints)
        if filter_string:
            params["filter"] = filter_string

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"eBay Search Request: URL={SEARCH_URL} PARAMS={params}")

        try:
            response = requests.get(
                SEARCH_URL,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                logger.error(f"eBay API error {response.status_code}: {response.text}")
                break
            
            data = response.json()
            page_items = data.get("itemSummaries", [])
            if not page_items:
                break

            items.extend(page_items)
            offset += page_size
        except Exception as e:
            logger.error(f"eBay API Request error: {e}")
            break
        
    return items

def get_category_aspects(category_id: str) -> List[str]:
    """Recupera l'elenco dei campi tecnici (Aspects) per una categoria via Taxonomy API."""
    token = _get_oauth_token()
    url = f"{TAXONOMY_URL}/category_tree/{EBAY_IT_TREE_ID}/get_item_aspect_metadata"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }
    params = {"category_id": category_id}
    
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            aspects = data.get("aspects", [])
            # Estraiamo i nomi degli aspetti obbligatori o rilevanti
            return [a["localizedAspectName"] for a in aspects if a.get("localizedAspectName")]
    except Exception as e:
        logger.error(f"Error fetching aspects for category {category_id}: {e}")
        
    return []
