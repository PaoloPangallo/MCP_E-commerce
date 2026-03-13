"""
Parser ottimizzato:
- spaCy lazy load
- brand vocab lazy load
- fast path rule-based
- fuzzy matching solo quando serve
- LLM opzionale e con timeout controllato
"""

from __future__ import annotations

import httpx
import json
import logging
import os
import re
import time
import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from rapidfuzz import fuzz, process
from app.db.redis import redis_client

# ============================================================
# LOAD .env
# ============================================================

try:
    from dotenv import load_dotenv

    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
    load_dotenv(override=False)
except Exception:
    pass

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

SPACY_MODEL = "it_core_news_sm"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "45"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "12"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

ENABLE_FUZZY_BRANDS = os.getenv("ENABLE_FUZZY_BRANDS", "true").strip().lower() == "true"

BRAND_WHITELIST = {
    "apple": "Apple",
    "iphone": "iPhone",
    "macbook": "MacBook",
    "samsung": "Samsung",
    "xiaomi": "Xiaomi",
    "huawei": "Huawei",
    "sony": "Sony",
    "nintendo": "Nintendo",
    "playstation": "PlayStation",
    "ps5": "PS5",
    "xbox": "Xbox",
    "lenovo": "Lenovo",
    "asus": "ASUS",
    "hp": "HP",
    "dell": "Dell",
    "acer": "Acer",
    "msi": "MSI",
    "lg": "LG",
    "dyson": "Dyson",
    "bose": "Bose",
    "jbl": "JBL",
}

CONDITION_SYNONYMS = {
    "new": {"nuovo", "nuova", "sigillato", "mai usato"},
    "used": {"usato", "usata", "seconda mano"},
    "refurbished": {"ricondizionato", "rigenerato", "refurbished"},
}

VAGUE_PRODUCT_TERMS = {
    "qualcosa",
    "qualcosa tipo",
    "cosa",
    "roba",
    "un qualcosa",
    "una cosa",
    "roba tipo",
    "prodotto",
}

DEFAULT_RESULT_TEMPLATE: Dict[str, Any] = {
    "original_query": "",
    "semantic_query": "",
    "product": None,
    "brands": [],
    "compatibilities": {},
    "constraints": [],
    "preferences": [],
    "_meta": {
        "llm_enabled": False,
        "llm_success": False,
        "llm_provider": None,
        "confidence": 0.0,
    },
}

NUM_PATTERN = r"(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)"
ARTICLE_PATTERN = r"(?:\s+(?:i|le|gli|ai|alle|agli|al|alla|di|a|un))?"

PRICE_RANGE_PATTERNS = [
    re.compile(
        rf"\b(?:tra|da){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\s+(?:e|a){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
    re.compile(
        rf"\b(?:prezzo|costo)\s+(?:minimo|min){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\s+(?:e|ed|a)\s+(?:prezzo|costo)?\s*(?:massimo|max){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
]

MAX_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:sotto|meno di|massimo|max|fino a|entro|non oltre){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
]

APPROX_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:circa|intorno a|intorno|sui|sulle|verso){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
]

MIN_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:almeno|minimo|min|sopra|oltre|più di){ARTICLE_PATTERN}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
]

EXPLICIT_PRICE_PATTERN = re.compile(
    rf"\b{NUM_PATTERN}\s*(?:euro|€)\b",
    re.IGNORECASE
)

# ============================================================
# LAZY LOADERS
# ============================================================

@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=1)
def load_brand_vocab() -> Tuple[str, ...]:
    try:
        root = Path(__file__).resolve().parents[2]
        file_path = root / "brand_vocab.json"

        if not file_path.exists():
            logger.warning("brand_vocab.json not found at %s", file_path)
            return tuple()

        with open(file_path, "r", encoding="utf-8") as f:
            brands = json.load(f)

        if not isinstance(brands, list):
            logger.warning("brand_vocab.json invalid format")
            return tuple()

        clean = []
        for b in brands:
            if isinstance(b, str):
                b = b.strip()
                if b:
                    clean.append(b)

        return tuple(sorted(set(clean)))

    except Exception as e:
        logger.warning("Failed loading brand vocab: %s", e)
        return tuple()


# ============================================================
# HELPERS
# ============================================================

def empty_result(original_query: str = "") -> Dict[str, Any]:
    result = deepcopy(DEFAULT_RESULT_TEMPLATE)
    result["original_query"] = original_query
    result["semantic_query"] = original_query
    return result


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("€", " euro ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_matching(text: str) -> str:
    return normalize_text(text).lower()


def normalize_brand(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw

    lowered = raw.lower()
    if lowered in BRAND_WHITELIST:
        return BRAND_WHITELIST[lowered]

    return raw[0].upper() + raw[1:] if len(raw) > 1 else raw.upper()


def normalize_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        s = value.strip().lower()
        s = s.replace("€", "").replace("euro", "").strip()

        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        else:
            if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
                s = s.replace(".", "")

        try:
            return float(s)
        except ValueError:
            return None

    return None


def dedupe_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []

    for item in items:
        key = item.lower() if isinstance(item, str) else json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


def is_vague_product(text: str) -> bool:
    t = (text or "").strip().lower()
    return not t or t in VAGUE_PRODUCT_TERMS or t.startswith("qualcosa")


def should_try_llm(query: str) -> bool:
    q = normalize_for_matching(query)

    # Fast path: if query is short and clearly parseable, avoid LLM if not needed
    easy_patterns = [
        r"\b(?:sotto|massimo|entro|tra|da|almeno|minimo|nuovo|usato|ricondizionato)\b",
        r"\b(?:iphone|samsung|xiaomi|ps5|playstation|macbook|lenovo|asus|dyson|bose|jbl)\b",
    ]

    hits = sum(1 for p in easy_patterns if re.search(p, q))
    return hits < 2


# ============================================================
# EXTRACTION
# ============================================================

def extract_base_price(text: str) -> Tuple[Optional[float], Optional[float]]:
    normalized = normalize_for_matching(text)

    for pattern in PRICE_RANGE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            p1 = normalize_float(m.group(1))
            p2 = normalize_float(m.group(2))
            if p1 is not None and p2 is not None:
                return min(p1, p2), max(p1, p2)

    for pattern in APPROX_PRICE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            val = normalize_float(m.group(1))
            if val is not None:
                return round(val * 0.8, 2), round(val * 1.2, 2)

    for pattern in MAX_PRICE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return None, normalize_float(m.group(1))

    for pattern in MIN_PRICE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return normalize_float(m.group(1)), None

    explicit_prices = [normalize_float(m.group(1)) for m in EXPLICIT_PRICE_PATTERN.finditer(normalized)]
    explicit_prices = [p for p in explicit_prices if p is not None]

    if len(explicit_prices) == 1:
        return None, explicit_prices[0]

    return None, None


def extract_condition(text: str) -> Optional[str]:
    normalized = normalize_for_matching(text)

    for canonical, variants in CONDITION_SYNONYMS.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", normalized):
                return canonical

    return None


def fuzzy_brand_detection(text: str, threshold: int = 88) -> List[str]:
    if not ENABLE_FUZZY_BRANDS:
        return []

    vocab = load_brand_vocab()
    if not vocab:
        return []

    words = re.findall(r"\b\w+\b", text.lower())
    found = []

    for word in words:
        if len(word) < 4:
            continue

        match = process.extractOne(
            word,
            cast(List[str], list(vocab)),
            scorer=fuzz.partial_ratio
        )

        if match:
            brand, score, _ = match
            if score >= threshold and abs(len(word) - len(brand)) <= 3:
                found.append(brand)

    return dedupe_keep_order(found)


def extract_brands(doc, original_text: str) -> List[str]:
    found: List[str] = []
    text_norm = normalize_for_matching(original_text)
    vocab = set(load_brand_vocab())

    for raw, canonical in BRAND_WHITELIST.items():
        if re.search(rf"\b{re.escape(raw)}\b", text_norm):
            found.append(canonical)

    # only fuzzy if whitelist found nothing
    if not found:
        found.extend(fuzzy_brand_detection(original_text))

    for ent in getattr(doc, "ents", []):
        candidate = ent.text.strip(" ,.-")
        if candidate and candidate in vocab:
            found.append(candidate)

    return dedupe_keep_order(found)


def extract_product(doc, original_text: str, brands: List[str]) -> Optional[str]:
    brand_norm = {b.lower() for b in brands}
    forbidden_tokens = {
        "prezzo", "euro", "minimo", "massimo", "costo",
        "sotto", "sopra", "circa", "intorno", "entro"
    }

    content_tokens: List[str] = []
    
    # Scansione lineare di tutti i token rilevanti
    for token in doc:
        # 1. Filtriamo punteggiatura e spazi
        if token.is_punct or token.is_space:
            continue

        # 2. Filtriamo i brand già identificati e le parole vietate
        t_low = token.text.lower()
        if t_low in brand_norm or t_low in forbidden_tokens:
            continue

        # 3. Prendiamo NOUN, PROPN, ADJ e NUM (fondamentali per le specifiche)
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}:
            content_tokens.append(token.text)

    if not content_tokens:
        return None

    # 4. Pulizia stop words solo se all'inizio o alla fine (es. "un", "con", "di")
    while content_tokens and get_nlp().vocab[content_tokens[0]].is_stop:
        content_tokens.pop(0)
    while content_tokens and get_nlp().vocab[content_tokens[-1]].is_stop:
        content_tokens.pop()
            
    if not content_tokens:
        return None
        
    # Uniamo fino a un massimo di 8 token (solitamente un prodotto non è più lungo)
    best = " ".join(content_tokens[:8]).strip()
    return None if is_vague_product(best) else best


# ============================================================
# RULE PARSE
# ============================================================

def rule_based_parse(query: str) -> Dict[str, Any]:
    normalized = normalize_text(query)
    doc = get_nlp()(normalized)

    brands = extract_brands(doc, query)
    min_price, max_price = extract_base_price(query)
    condition = extract_condition(query)
    product = extract_product(doc, query, brands)

    result = empty_result(query)
    result["brands"] = brands
    result["product"] = product
    result["semantic_query"] = query

    if min_price is not None and max_price is not None:
        result["constraints"].append({"type": "price", "operator": "between", "value": [min_price, max_price]})
    elif max_price is not None:
        result["constraints"].append({"type": "price", "operator": "<=", "value": max_price})
    elif min_price is not None:
        result["constraints"].append({"type": "price", "operator": ">=", "value": min_price})

    if condition:
        result["constraints"].append({"type": "condition", "value": condition})

    return result


# ============================================================
# LLM CALLS
# ============================================================

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"


async def call_ollama(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    think: bool = False,
) -> Optional[str]:
    """
    Wrapper async per Ollama chat API.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 768,
            "num_ctx": 8192,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=float(OLLAMA_TIMEOUT))) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()

        if content:
            return content

        if think and thinking:
            return await call_ollama(prompt, system_prompt=system_prompt, think=False)

        return None

    except Exception as e:
        logger.warning("Ollama async error: %s", e)
        return None


async def call_gemini(prompt: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=float(GEMINI_TIMEOUT)) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                return None
            data = response.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            return text.strip() if text else None
    except Exception as e:
        logger.warning("Gemini async error: %s", e)
        return None


async def call_llm(prompt: str) -> Tuple[Optional[str], str]:
    primary = LLM_PROVIDER
    fallback = LLM_FALLBACK_PROVIDER

    async def _call(provider: str) -> Optional[str]:
        if provider == "ollama":
            return await call_ollama(prompt)
        if provider == "gemini":
            return await call_gemini(prompt)
        return None

    out = await _call(primary)
    if out:
        return out, primary

    if fallback and fallback != primary:
        out2 = await _call(fallback)
        if out2:
            return out2, fallback

    return None, primary


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_condition(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value_norm = str(value).strip().lower()
    if not value_norm:
        return None

    for canonical, variants in CONDITION_SYNONYMS.items():
        if value_norm == canonical or value_norm in variants:
            return canonical

    return value_norm


def normalize_price_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    operator = item.get("operator")
    value = item.get("value")

    if operator not in {"<=", ">=", "between"}:
        return None

    if operator == "between":
        if not isinstance(value, list) or len(value) != 2:
            return None

        v1 = normalize_float(value[0])
        v2 = normalize_float(value[1])

        if v1 is None or v2 is None:
            return None

        return {"type": "price", "operator": "between", "value": [min(v1, v2), max(v1, v2)]}

    v = normalize_float(value)
    if v is None:
        return None

    return {"type": "price", "operator": operator, "value": v}


def normalize_condition_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    val = normalize_condition(item.get("value"))
    if not val:
        return None
    return {"type": "condition", "value": val}


def normalize_constraint(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(c, dict):
        return None

    ctype = c.get("type")
    if ctype == "price":
        return normalize_price_item(c)
    if ctype == "condition":
        return normalize_condition_item(c)

    return None


def normalize_preference(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(p, dict):
        return None

    ptype = p.get("type")

    if ptype == "price":
        return normalize_price_item(p)

    if ptype == "condition":
        return normalize_condition_item(p)

    if ptype == "brand":
        val = p.get("value")
        if not isinstance(val, str) or not val.strip():
            return None
        return {"type": "brand", "value": normalize_brand(val)}

    return None


def validate_llm_result(data: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    result = empty_result(original_query)

    semantic_query = data.get("semantic_query")
    result["semantic_query"] = (
        semantic_query.strip()
        if isinstance(semantic_query, str) and semantic_query.strip()
        else original_query
    )

    brands = data.get("brands", [])
    if isinstance(brands, list):
        normalized_brands = []
        for b in brands:
            if isinstance(b, str) and b.strip():
                normalized_brands.append(normalize_brand(b))
        result["brands"] = dedupe_keep_order(normalized_brands)

    product = data.get("product")
    if isinstance(product, str) and product.strip().lower() not in {"", "null", "none"}:
        cleaned = product.strip()
        result["product"] = None if is_vague_product(cleaned) else cleaned

    constraints = data.get("constraints", [])
    if isinstance(constraints, list):
        norm_constraints = []
        for c in constraints:
            nc = normalize_constraint(c)
            if nc is not None:
                norm_constraints.append(nc)
        result["constraints"] = dedupe_keep_order(norm_constraints)

    preferences = data.get("preferences", [])
    if isinstance(preferences, list):
        norm_prefs = []
        for p in preferences:
            np = normalize_preference(p)
            if np is not None:
                norm_prefs.append(np)
        result["preferences"] = dedupe_keep_order(norm_prefs)

    compatibilities = data.get("compatibilities", {})
    if isinstance(compatibilities, dict):
        result["compatibilities"] = {
            str(k): str(v)
            for k, v in compatibilities.items()
            if str(k).strip() and str(v).strip()
        }

    return result


def enforce_numeric_consistency(original_query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    original_numbers = re.findall(r"\b\d+\b", original_query)
    product = result.get("product")

    if not product:
        return result

    product_numbers = re.findall(r"\b\d+\b", product)
    if product_numbers and original_numbers:
        if not any(n in original_numbers for n in product_numbers):
            result["product"] = None
            result["semantic_query"] = original_query

    return result


# ============================================================
# LLM PARSE
# ============================================================

async def llm_parse(query: str) -> Tuple[Optional[Dict[str, Any]], str]:
    prompt = f"""
You are a strict semantic query parser for an e-commerce assistant.

Return ONLY valid minified JSON.
No markdown.
No explanations.
No code fences.

Schema:
{{
  "semantic_query": "",
  "product": null,
  "brands": [],
  "compatibilities": {{}},
  "constraints": [],
  "preferences": []
}}

Allowed constraint/preference item types:

Price:
{{
  "type": "price",
  "operator": "<=" | ">=" | "between",
  "value": number OR [min, max]
}}

Condition:
{{
  "type": "condition",
  "value": "new" | "used" | "refurbished"
}}

Optional brand preference:
{{
  "type": "brand",
  "value": "Samsung"
}}

Rules:
- Put ONLY mandatory requirements in constraints.
- Put optional wishes in preferences.
- Do NOT invent brands or products.
- If max budget is given, use only "<=".
- If min budget is given, use only ">=".
- If min and max are given, use "between".
- If approximate budget is given, use +/-20% with "between".
- Output JSON only.

Query: {json.dumps(query, ensure_ascii=False)}
""".strip()

    response, used_provider = await call_llm(prompt)
    if not response:
        return None, used_provider

    json_text = extract_first_json_object(response)
    if not json_text:
        return None, used_provider

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError:
        return None, used_provider

    if not isinstance(raw, dict):
        return None, used_provider

    parsed = validate_llm_result(raw, query)
    parsed = enforce_numeric_consistency(query, parsed)
    return parsed, used_provider


# ============================================================
# MERGE / CONFIDENCE
# ============================================================

def merge_results(rule_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm_result:
        semantic_parts = []

        if rule_result.get("brands"):
            semantic_parts.extend(rule_result["brands"])

        if rule_result.get("product"):
            semantic_parts.append(rule_result["product"])

        rule_result["semantic_query"] = (
            " ".join(semantic_parts)
            if semantic_parts
            else rule_result["original_query"]
        )
        return rule_result

    final = empty_result(rule_result["original_query"])
    final["semantic_query"] = llm_result.get("semantic_query") or rule_result.get("semantic_query") or rule_result["original_query"]
    final["product"] = llm_result.get("product")

    llm_brands = llm_result.get("brands", []) or []
    rule_brands = rule_result.get("brands", []) or []
    final["brands"] = dedupe_keep_order(llm_brands + rule_brands)

    rule_constraints = rule_result.get("constraints", [])
    llm_constraints = llm_result.get("constraints", [])

    llm_price_constraints = [c for c in llm_constraints if c.get("type") == "price"]
    price_constraints = llm_price_constraints if llm_price_constraints else [c for c in rule_constraints if c.get("type") == "price"]

    llm_condition = next((c for c in llm_constraints if c.get("type") == "condition"), None)
    condition_constraints = [llm_condition] if llm_condition else [c for c in rule_constraints if c.get("type") == "condition"]

    final["constraints"] = price_constraints + condition_constraints
    final["preferences"] = llm_result.get("preferences", []) or []
    final["compatibilities"] = llm_result.get("compatibilities", {}) or {}

    return final


def compute_confidence(final_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> float:
    score = 0.15

    if final_result.get("brands"):
        score += 0.15
    if final_result.get("constraints"):
        score += 0.20
    if final_result.get("product"):
        score += 0.15
    if final_result.get("preferences"):
        score += 0.05
    if llm_result is not None:
        score += 0.10

    return round(min(score, 0.95), 2)


def correct_brands_in_text(text: str) -> str:
    vocab = load_brand_vocab()
    if not vocab or not ENABLE_FUZZY_BRANDS:
        return text

    words = text.split()
    corrected = []

    for w in words:
        w_low = w.lower()
        
        # SKIP numeric/postal codes
        if re.fullmatch(r"\d+", w):
            corrected.append(w)
            continue
            
        # EXACT MATCH check - if it's already a known brand, don't fuzzy replace it
        if w_low in BRAND_WHITELIST:
            corrected.append(BRAND_WHITELIST[w_low])
            continue

        if len(w) < 4:
            corrected.append(w)
            continue

        match = process.extractOne(
            w,
            cast(List[str], list(vocab)),
            scorer=fuzz.token_sort_ratio # Stricter than partial_ratio for full words
        )

        if match:
            brand, score, _ = match
            # Only replace if score is very high and it's not a short word match
            if score >= 92 and abs(len(w) - len(brand)) <= 2:
                corrected.append(brand)
                continue

        corrected.append(w)

    return " ".join(corrected)


# ============================================================
# MAIN SERVICE
# ============================================================

async def parse_query_service(
    text: str,
    use_llm: bool = True,
    include_meta: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    text = normalize_text(text)
    text = correct_brands_in_text(text)

    # Caching check
    cache_key = f"query_parse:{text}:{use_llm}"
    if use_llm:
        cached = redis_client.get_json(cache_key)
        if cached:
            logger.info("Parser cache hit for query: %s", text)
            return cached

    rule_result = rule_based_parse(text)
    llm_result = None
    used_provider: Optional[str] = None

    effective_use_llm = use_llm and should_try_llm(text)

    if effective_use_llm:
        parsed_llm, used_provider = await llm_parse(text)
        llm_result = parsed_llm

    final = merge_results(rule_result, llm_result)

    if include_meta:
        final["_meta"] = {
            "llm_enabled": effective_use_llm,
            "llm_success": llm_result is not None,
            "llm_provider": used_provider if effective_use_llm else None,
            "confidence": compute_confidence(final, llm_result),
        }
    else:
        final.pop("_meta", None)

    # Cache the result if LLM was used and successful or if rule-based was enough
    if effective_use_llm and llm_result:
        redis_client.set_json(cache_key, final, ttl_seconds=3600) # 1 hour cache

    return final