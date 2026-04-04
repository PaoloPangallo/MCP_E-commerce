"""
services/parser.py — Query parser ottimizzato
----------------------------------------------
Responsabilità:
  - Parsing rule-based (spaCy + regex)
  - Parsing LLM-enhanced (via app.llm.client — solo Ollama Cloud)
  - Merge e normalizzazione risultati

NON contiene più:
  - client HTTP Ollama/Gemini (→ app/llm/client.py)
  - Vision (→ app/llm/vision.py)
"""
from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from rapidfuzz import fuzz, process

from app.db.redis import redis_client
from app.llm.client import call_llm
from app.utils.collections import dedupe_keep_order

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

SPACY_MODEL = "it_core_news_sm"

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
    "levis": "Levi's",
    "levi": "Levi's",
    "nike": "Nike",
    "adidas": "Adidas",
    "puma": "Puma",
    "reebok": "Reebok",
    "zara": "Zara",
    "bimby": "Bimby",
    "folletto": "Folletto",
}

STOP_WORDS = {
    "ciao", "buon", "buongiorno", "buonasera", "ehi", "hey",
    "stavo", "cercando", "cerco", "trovami", "mostrami", "fammi", "vedere",
    "per", "favore", "piacere", "grazie", "grazie mille",
    "un", "una", "uno", "dei", "degli", "delle", "il", "lo", "la", "i", "gli", "le",
}

CONDITION_SYNONYMS = {
    "new": {"nuovo", "nuova", "sigillato", "mai usato"},
    "used": {"usato", "usata", "seconda mano"},
    "refurbished": {"ricondizionato", "rigenerato", "refurbished"},
}

VAGUE_PRODUCT_TERMS = {
    "qualcosa", "qualcosa tipo", "cosa", "roba",
    "un qualcosa", "una cosa", "roba tipo", "prodotto",
}

DEFAULT_RESULT_TEMPLATE: Dict[str, Any] = {
    "original_query": "",
    "semantic_query": "",
    "product": None,
    "product_type": "Unknown",
    "brands": [],
    "compatibilities": {},
    "constraints": [],
    "preferences": [],
    "exclusions": [],
    "target_seller": None,
    "_meta": {
        "llm_enabled": False,
        "llm_success": False,
        "llm_provider": None,
        "confidence": 0.0,
    },
}

# Le regex complesse per i prezzi sono state rimosse. Demandiamo l'estrazione semantica interamente al LLM.

# ── Lazy loaders ─────────────────────────────────────────────────────────────


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

        clean = [b.strip() for b in brands if isinstance(b, str) and b.strip()]
        return tuple(sorted(set(clean)))

    except Exception as exc:
        logger.warning("Failed loading brand vocab: %s", exc)
        return tuple()


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def is_vague_product(text: str) -> bool:
    t = (text or "").strip().lower()
    return not t or t in VAGUE_PRODUCT_TERMS or t.startswith("qualcosa")


def should_try_llm(query: str) -> bool:
    return True # LLM è ora il motore principale di estrazione semantica.


# Funzioni euristico/regex (extract_base_price, extract_condition, extract_product, rule_based_parse) rimosse.


# ── JSON helpers ──────────────────────────────────────────────────────────────

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


# ── Normalizers ───────────────────────────────────────────────────────────────

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
    if ctype == "aspect":
        name = c.get("name")
        value = c.get("value")
        if name and value:
            return {"type": "aspect", "name": str(name), "value": str(value)}
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
        normalized_brands = [normalize_brand(b) for b in brands if isinstance(b, str) and b.strip()]
        result["brands"] = dedupe_keep_order(normalized_brands)

    product = data.get("product")
    if isinstance(product, str) and product.strip().lower() not in {"", "null", "none"}:
        cleaned = product.strip()
        result["product"] = None if is_vague_product(cleaned) else cleaned

    product_type = data.get("product_type")
    if product_type in {"Main", "Accessory", "Part", "Service", "Unknown"}:
        result["product_type"] = product_type

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
            np_ = normalize_preference(p)
            if np_ is not None:
                norm_prefs.append(np_)
        result["preferences"] = dedupe_keep_order(norm_prefs)

    exclusions = data.get("exclusions", [])
    if isinstance(exclusions, list):
        result["exclusions"] = dedupe_keep_order([str(e).strip().lower() for e in exclusions if str(e).strip()])

    compatibilities = data.get("compatibilities", {})
    if isinstance(compatibilities, dict):
        result["compatibilities"] = {
            str(k): str(v)
            for k, v in compatibilities.items()
            if str(k).strip() and str(v).strip()
        }
    
    target_seller = data.get("target_seller")
    if isinstance(target_seller, str) and target_seller.strip().lower() not in {"", "null", "none"}:
        result["target_seller"] = target_seller.strip()

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


# ── LLM parse ─────────────────────────────────────────────────────────────────

async def llm_parse(query: str, context_info: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    context_block = ""
    if context_info:
        context_block = f"\nCONVERSATION CONTEXT (History of recent topics/products):\n{context_info}\n"

    prompt = f"""
You are a strict semantic query parser for an e-commerce assistant.
{context_block}
Return ONLY valid minified JSON.
No markdown. No explanations. No code fences.

Schema:
{{
  "semantic_query": "",
  "product": null,
  "product_type": "Main" | "Accessory" | "Part" | "Service" | "Unknown",
  "brands": [],
  "compatibilities": {{}},
  "constraints": [],
  "preferences": [],
  "exclusions": [],
  "target_seller": null
}}

Allowed constraint/preference item types:

Price:
{{"type": "price", "operator": "<=" | ">=" | "between", "value": number OR [min, max]}}

Condition:
{{"type": "condition", "value": "new" | "used" | "refurbished"}}

Optional brand preference:
{{"type": "brand", "value": "Samsung"}}

Specific attribute/aspect (color, material, storage, size, model_version, etc.):
{{"type": "aspect", "name": "Color" | "Material" | "Storage" | "Size" | "Model", "value": "Blue" | "Leather" | "256GB" | "42" | "iPhone 13"}}

Rules:
- Put ONLY mandatory requirements in constraints.
- Put optional wishes in preferences.
- Extract adjectives like "blu", "rosa", "leather", "taglia 42", "64gb" as 'aspect' constraints.
- 'product_type': If the user wants the device itself (phone, laptop, console), use "Main". If they want a case, cover, charger, use "Accessory". If they want a screen replacement, battery, scocca, use "Part".
- Do NOT invent brands or products.
- Resolve pronouns (e.g., "li", "le", "quelli", "cercalo", "them", "it") using the CONVERSATION CONTEXT.
- IMPORTANT: If the user says "without", "no", "senza", "non", "nò" followed by a feature (e.g., "senza zip", "no wireless", "non usato"), extract those features into the "exclusions" array.
- If the user says "cercalo", "trovali", "show me more", etc., the 'product' and 'brands' fields must be populated based on the previous topic found in the context.
- Target Seller: Extract the seller name if the user mentions one (e.g., "venduto da pegaso_italia", "di apple_store", "da monclick").
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


# merge_results non serve più in quanto usiamo solo LLM


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

        if re.fullmatch(r"\d+", w):
            corrected.append(w)
            continue

        if w_low in BRAND_WHITELIST:
            corrected.append(BRAND_WHITELIST[w_low])
            continue

        if len(w) < 4:
            corrected.append(w)
            continue

        match = process.extractOne(
            w,
            cast(List[str], list(vocab)),
            scorer=fuzz.token_sort_ratio,
        )
        if match:
            brand, score, _ = match
            if score >= 92 and abs(len(w) - len(brand)) <= 2:
                corrected.append(brand)
                continue

        corrected.append(w)

    return " ".join(corrected)


# ── Main service ──────────────────────────────────────────────────────────────

async def parse_query_service(
    text: str,
    use_llm: bool = True,
    include_meta: bool = True,
    context_info: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    text = normalize_text(text)
    text = correct_brands_in_text(text)

    cache_key = f"query_parse:{text}:{use_llm}:{context_info or ''}"
    if use_llm:
        cached = redis_client.get_json(cache_key)
        if cached:
            logger.info("Parser cache hit for query: %s", text)
            return cached

    llm_result = None
    used_provider: Optional[str] = None

    if use_llm:
        llm_result, used_provider = await llm_parse(text, context_info=context_info)

    final = llm_result if llm_result else empty_result(text)

    if include_meta:
        final["_meta"] = {
            "llm_enabled": use_llm,
            "llm_success": llm_result is not None,
            "llm_provider": used_provider if use_llm else None,
            "confidence": 0.95 if llm_result else 0.1,
        }
    else:
        final.pop("_meta", None)

    if use_llm and llm_result:
        redis_client.set_json(cache_key, final, ttl_seconds=3600)

    return final