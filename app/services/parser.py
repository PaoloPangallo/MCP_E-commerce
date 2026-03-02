"""
app/services/parser.py

Parser NLP "neutro" e coerente con quanto deciso:
- NESSUNA logica di business / intent (te la gestisci tu fuori)
- Output stabile: semantic_query, product, brands, constraints, preferences, _meta
- Parsing minimalista ma utile per retrieval: brand/product/price/condition
- LLM come componente principale + fallback rule-based
- Validazione/normalizzazione robusta

✅ LLM Provider switch (OLLAMA / GEMINI) via .env:
  - LLM_PROVIDER=ollama | gemini
  - GEMINI_API_KEY=...
"""

import json
import logging
import os
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import spacy
from rapidfuzz import fuzz, process

# ============================================================
# LOAD .env (optional)
# ============================================================
try:
    from dotenv import load_dotenv  # pip install python-dotenv

    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
    load_dotenv(override=False)
except Exception:
    pass

# ============================================================
# CONFIG
# ============================================================

SPACY_MODEL = "it_core_news_sm"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Whitelist SOLO per canonicalizzazione fallback (non deve essere “la fonte” dei brand)
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
}

DEFAULT_RESULT_TEMPLATE: Dict[str, Any] = {
    "original_query": "",
    "semantic_query": "",
    "product": None,
    "brands": [],
    "constraints": [],
    "preferences": [],
    "_meta": {
        "llm_enabled": False,
        "llm_success": False,
        "llm_provider": None,
        "confidence": 0.0,
    },
}

# ============================================================
# LOAD NLP
# ============================================================

def load_nlp(model_name: str = SPACY_MODEL):
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"Modello spaCy '{model_name}' non trovato. "
            f"Installa con: python -m spacy download {model_name}"
        ) from e


nlp = load_nlp()

# ============================================================
# BRAND VOCAB LOAD (offline da brand_vocab.json)
# ============================================================

BRAND_VOCAB: List[str] = []
_BRAND_CANON_BY_LOWER: Dict[str, str] = {}
_BRAND_LOWER_LIST: List[str] = []

def _rebuild_brand_indexes() -> None:
    global _BRAND_CANON_BY_LOWER, _BRAND_LOWER_LIST
    _BRAND_CANON_BY_LOWER = {b.lower(): b for b in BRAND_VOCAB if isinstance(b, str) and b.strip()}
    _BRAND_LOWER_LIST = list(_BRAND_CANON_BY_LOWER.keys())

def load_brand_vocab() -> List[str]:
    try:
        # stessa directory del parser
        file_path = Path(__file__).resolve().parent / "brand_vocab.json"

        print("Loading brand vocab from:", file_path)

        if not file_path.exists():
            logger.warning("brand_vocab.json not found at %s", file_path)
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            brands = json.load(f)

        if not isinstance(brands, list):
            logger.warning("brand_vocab.json invalid format (expected list)")
            return []

        clean = [b.strip() for b in brands if isinstance(b, str) and b.strip()]

        logger.info("Loaded %d brands into BRAND_VOCAB", len(clean))
        return sorted(set(clean))

    except Exception as e:
        logger.warning("Failed loading brand_vocab.json: %s", e)
        return []


BRAND_VOCAB = load_brand_vocab()
_rebuild_brand_indexes()

def update_brand_vocab(brands: List[str]) -> None:
    """Opzionale: se vuoi arricchire runtime. Manteniamo gli indici coerenti."""
    global BRAND_VOCAB
    changed = False
    for b in brands:
        if isinstance(b, str):
            b = b.strip()
            if b and b not in BRAND_VOCAB:
                BRAND_VOCAB.append(b)
                changed = True
    if changed:
        BRAND_VOCAB = sorted(set(BRAND_VOCAB))
        _rebuild_brand_indexes()

def clean_semantic_query(text: str) -> str:
    if not text:
        return text

    text = re.sub(
        r"\b(intorno\s+ai|intorno\s+a|intorno\s+alle|circa|sui|sulle|verso|"
        r"fino\s+a|massimo|max|meno\s+di|oltre|almeno|minimo|min|"
        r"sotto|sopra|più\s+di|tra|da)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\b(euro|eur|€)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(i|gli|le|il|lo|la|un|una|ai|alle|al)\b", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# UTILS
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
        if isinstance(item, str):
            key = item.lower()
        else:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

def is_vague_product(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in VAGUE_PRODUCT_TERMS:
        return True
    if t.startswith("qualcosa"):
        return True
    return False

def _valid_brand_set() -> set:
    # brand_vocab è la fonte. la whitelist è solo fallback canonico.
    return set(BRAND_VOCAB) | set(BRAND_WHITELIST.values())

def filter_valid_brands(brands: List[str]) -> List[str]:
    valid = _valid_brand_set()
    cleaned: List[str] = []

    for b in brands:
        if not isinstance(b, str):
            continue

        b = b.strip()
        if not b:
            continue

        # ❌ escludi parole generiche comuni
        if b.lower() in {"scarpa", "scarpe", "euro", "uomo", "donna"}:
            continue

        if b in valid:
            cleaned.append(b)
            continue

        canon = _BRAND_CANON_BY_LOWER.get(b.lower())
        if canon:
            cleaned.append(canon)

    return dedupe_keep_order(cleaned)

# ============================================================
# PRICE EXTRACTION
# ============================================================

NUM_PATTERN = r"(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)"
OPT_ART = r"(?:\s+(?:i|gli|le))?"

PRICE_RANGE_PATTERNS = [
    re.compile(
        rf"\b(?:tra|da){OPT_ART}\s+{NUM_PATTERN}\s*(?:euro|€)?\s+(?:e|a){OPT_ART}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE,
    ),
]

MAX_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:sotto|meno\s+di|massimo|max|fino\s+a|entro|non\s+oltre){OPT_ART}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE,
    ),
]

MIN_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:almeno|minimo|min|sopra|oltre|più\s+di){OPT_ART}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE,
    ),
]

APPROX_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:circa|intorno\s+a|intorno\s+ai|intorno\s+alle|sui|sulle|verso){OPT_ART}\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE,
    ),
]

EXPLICIT_PRICE_PATTERN = re.compile(
    rf"\b{NUM_PATTERN}\s*(?:euro|€)\b",
    re.IGNORECASE,
)

def extract_price_constraint(text: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_for_matching(text)

    min_price = None
    max_price = None

    # RANGE ESPLICITO
    for pattern in PRICE_RANGE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            p1 = normalize_float(m.group(1))
            p2 = normalize_float(m.group(2))
            if p1 is not None and p2 is not None:
                return {
                    "type": "price",
                    "operator": "between",
                    "value": [min(p1, p2), max(p1, p2)],
                }

    # MIN
    for pattern in MIN_PRICE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            val = normalize_float(m.group(1))
            if val is not None:
                min_price = val

    # MAX
    for pattern in MAX_PRICE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            val = normalize_float(m.group(1))
            if val is not None:
                max_price = val

    # Se entrambi trovati → between
    if min_price is not None and max_price is not None:
        return {
            "type": "price",
            "operator": "between",
            "value": [min(min_price, max_price), max(min_price, max_price)],
        }

    if min_price is not None:
        return {
            "type": "price",
            "operator": ">=",
            "value": min_price,
        }

    if max_price is not None:
        return {
            "type": "price",
            "operator": "<=",
            "value": max_price,
        }

    return None

# ============================================================
# CONDITION EXTRACTION
# ============================================================

def extract_condition(text: str) -> Optional[str]:
    normalized = normalize_for_matching(text)
    for canonical, variants in CONDITION_SYNONYMS.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", normalized):
                return canonical
    return None

# ============================================================
# BRAND EXTRACTION / CORRECTION
# ============================================================

def _fuzzy_match_brand(token: str, threshold: int = 80) -> Optional[str]:
    """Match su brand_vocab in modo case-insensitive e ritorna brand canonico."""
    if not _BRAND_LOWER_LIST:
        return None

    t = token.strip()
    if len(t) < 4:
        return None

    t_low = t.lower()

    match = process.extractOne(t_low, _BRAND_LOWER_LIST, scorer=fuzz.ratio)
    if not match:
        return None

    best_low, score, _ = match
    if score < threshold:
        return None

    canon = _BRAND_CANON_BY_LOWER.get(best_low)
    if not canon:
        return None

    # vincolo generico: stessa iniziale (riduce falsi positivi senza hardcode)
    if t_low[0] != canon.lower()[0]:
        return None

    # vincolo generico: lunghezze compatibili
    if abs(len(t_low) - len(canon)) > 3:
        return None

    return canon

def fuzzy_brand_detection(text: str, threshold: int = 80) -> List[str]:
    words = re.findall(r"\b\w+\b", text)
    found: List[str] = []
    for w in words:
        b = _fuzzy_match_brand(w, threshold=threshold)
        if b:
            found.append(b)
    return dedupe_keep_order(found)

def correct_brands_in_text(text: str) -> str:
    """Corregge token brand-like nel testo usando brand_vocab (no hardcode)."""
    if not BRAND_VOCAB:
        return text

    tokens = re.findall(r"\w+|\S", text)
    corrected: List[str] = []

    for tok in tokens:
        if not tok.isalpha() or len(tok) < 4:
            corrected.append(tok)
            continue

        canon = _fuzzy_match_brand(tok, threshold=80)
        corrected.append(canon if canon else tok)

    # ricostruzione con spazi
    out = ""
    for t in corrected:
        if re.fullmatch(r"\W", t):
            out += t
        else:
            out += (" " if out and not out.endswith((" ", "\n")) else "") + t
    return out.strip()

def extract_brands(doc, original_text: str) -> List[str]:
    """Brand = whitelist match + fuzzy su vocab + (opzionale) NER validato su vocab."""
    found: List[str] = []
    text_norm = normalize_for_matching(original_text)

    # 1) whitelist (solo canonicalizzazione)
    for raw, canonical in BRAND_WHITELIST.items():
        if re.search(rf"\b{re.escape(raw)}\b", text_norm):
            found.append(canonical)

    logger.debug("Brand detection on: %s", original_text)

    # 2) fuzzy su vocab reale
    found.extend(fuzzy_brand_detection(original_text, threshold=80))

    # 3) NER: teniamo SOLO se è brand in vocab (nessuna euristica extra)
    for ent in getattr(doc, "ents", []):
        cand = ent.text.strip(" ,.-")
        if not cand:
            continue
        canon = _BRAND_CANON_BY_LOWER.get(cand.lower())
        if canon:
            found.append(canon)

    return filter_valid_brands(dedupe_keep_order(found))

# ============================================================
# PRODUCT EXTRACTION (fallback conservativo)
# ============================================================

def extract_product(doc, original_text: str, brands: List[str]) -> Optional[str]:
    brand_norm = {b.lower() for b in brands}
    candidates: List[str] = []

    try:
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if not text:
                continue

            text_norm = text.lower()

            # ❌ evita chunk che contengono brand
            if any(b in text_norm for b in brand_norm):
                continue

            # ❌ evita numeri/prezzi
            if re.search(r"\d+\s*(euro|€)?", text_norm):
                continue

            # ❌ evita parole monetarie
            if re.search(r"\b(euro|eur)\b", text_norm):
                continue

            tokens = [
                t.text for t in chunk
                if not t.is_punct
                and not t.is_space
                and t.text.lower() not in {"euro", "eur"}
            ]

            if tokens:
                cand = " ".join(tokens).strip()
                if cand and not is_vague_product(cand):
                    candidates.append(cand)
    except Exception:
        pass

    if candidates:
        candidates = sorted(candidates, key=len, reverse=True)
        best = candidates[0].strip()
        return None if is_vague_product(best) else best

    # fallback token-level
    content_tokens: List[str] = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        if token.text.lower() in brand_norm:
            continue

        if token.text.lower() in {"euro", "eur"}:
            continue

        if re.fullmatch(r"\d+(?:[.,]\d+)?", token.text):
            continue

        if token.pos_ in {"NOUN", "PROPN"}:
            content_tokens.append(token.text)

    if content_tokens:
        best = " ".join(content_tokens[:3]).strip()
        return None if is_vague_product(best) else best

    return None

# ============================================================
# RULE-BASED PARSER
# ============================================================

def rule_based_parse(query: str) -> Dict[str, Any]:
    normalized = normalize_text(query)
    doc = nlp(normalized)

    brands = extract_brands(doc, query)
    condition = extract_condition(query)
    product = extract_product(doc, query, brands)
    price_c = extract_price_constraint(query)

    result = empty_result(query)
    result["brands"] = brands
    result["product"] = product
    result["semantic_query"] = query  # LLM/cleaner può migliorare

    if price_c:
        result["constraints"].append(price_c)

    if condition:
        result["constraints"].append({"type": "condition", "value": condition})

    return result

# ============================================================
# LLM PROVIDERS
# ============================================================

def call_ollama(prompt: str) -> Optional[str]:
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 350},
        }

        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        if response.status_code != 200:
            logger.warning("Ollama HTTP %s", response.status_code)
            return None

        data = response.json()
        return data.get("response", "").strip() or None

    except Exception as e:
        logger.exception("Ollama REST error: %s", e)
        return None

def call_gemini(prompt: str) -> Optional[str]:
    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] GEMINI call started", request_id)

    if not GEMINI_API_KEY:
        logger.error("[%s] GEMINI_API_KEY missing", request_id)
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 350},
    }

    try:
        t0 = time.time()
        response = requests.post(url, json=payload, timeout=GEMINI_TIMEOUT)
        elapsed = round(time.time() - t0, 2)
        logger.info("[%s] Gemini HTTP %s in %ss", request_id, response.status_code, elapsed)

        if response.status_code == 429:
            logger.warning("[%s] Gemini rate limit (429)", request_id)
            return None
        if response.status_code != 200:
            logger.error("[%s] Gemini error body: %s", request_id, response.text[:300])
            return None

        data = response.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )
        return text.strip() if text else None

    except requests.Timeout:
        logger.error("[%s] Gemini TIMEOUT after %ss", request_id, GEMINI_TIMEOUT)
        return None
    except Exception as e:
        logger.exception("[%s] Gemini unexpected error: %s", request_id, e)
        return None

def call_llm(prompt: str, provider: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Esegue chiamata LLM usando:
    - provider passato esplicitamente
    - altrimenti LLM_PROVIDER da env
    """
    primary = (provider or LLM_PROVIDER).strip().lower()
    fallback = LLM_FALLBACK_PROVIDER.strip().lower()

    def _call(p: str) -> Optional[str]:
        if p == "ollama":
            return call_ollama(prompt)
        if p == "gemini":
            return call_gemini(prompt)
        logger.error("Unknown provider: %s", p)
        return None

    out = _call(primary)
    if out:
        return out, primary

    if fallback and fallback != primary:
        out2 = _call(fallback)
        if out2:
            return out2, fallback

    return None, primary

# ============================================================
# JSON EXTRACTION
# ============================================================

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
                return text[start : i + 1]
    return None

# ============================================================
# VALIDATION / NORMALIZATION
# ============================================================

_ALLOWED_PRICE_OPS = {"<=", ">=", "between", "approx"}

def normalize_price_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    operator = item.get("operator")
    value = item.get("value")

    if operator not in _ALLOWED_PRICE_OPS:
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
        # brand preference valido solo se in vocab/whitelist
        canon = _BRAND_CANON_BY_LOWER.get(val.strip().lower())
        if canon:
            return {"type": "brand", "value": canon}
        norm = normalize_brand(val)
        if norm in _valid_brand_set():
            return {"type": "brand", "value": norm}
        return None

    if ptype == "sort":
        val = p.get("value")
        if val in {"price_asc", "price_desc", "newest"}:
            return {"type": "sort", "value": val}
        return None

    return None

def validate_llm_result(data: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    result = empty_result(original_query)

    semantic_query = data.get("semantic_query")
    result["semantic_query"] = (
        semantic_query.strip()
        if isinstance(semantic_query, str) and semantic_query.strip()
        else original_query
    )

    # Brands: accetta solo quelli validi (vocab/whitelist), anche se LLM spara parole
    brands = data.get("brands", [])
    normalized_brands: List[str] = []
    if isinstance(brands, list):
        for b in brands:
            if isinstance(b, str) and b.strip():
                canon = _BRAND_CANON_BY_LOWER.get(b.strip().lower())
                if canon:
                    normalized_brands.append(canon)
                else:
                    nb = normalize_brand(b)
                    if nb in _valid_brand_set():
                        normalized_brands.append(nb)
    result["brands"] = filter_valid_brands(dedupe_keep_order(normalized_brands))

    product = data.get("product")
    if isinstance(product, str) and product.strip().lower() not in {"", "null", "none"}:
        cleaned = product.strip()
        result["product"] = None if is_vague_product(cleaned) else cleaned
    else:
        result["product"] = None

    constraints = data.get("constraints", [])
    if isinstance(constraints, list):
        norm_constraints: List[Dict[str, Any]] = []
        for c in constraints:
            nc = normalize_constraint(c)
            if nc is not None:
                norm_constraints.append(nc)
        result["constraints"] = dedupe_keep_order(norm_constraints)

    preferences = data.get("preferences", [])
    if isinstance(preferences, list):
        norm_prefs: List[Dict[str, Any]] = []
        for p in preferences:
            np = normalize_preference(p)
            if np is not None:
                norm_prefs.append(np)
        result["preferences"] = dedupe_keep_order(norm_prefs)

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

def llm_parse(query: str, provider: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    prompt = f"""
    You are a strict semantic query parser for an Italian e-commerce assistant.

    Your task:
    Extract structured data from a user query.

    Return ONLY valid minified JSON.
    No markdown.
    No explanations.
    No code fences.

    Schema:
    {{
      "semantic_query": "",
      "product": null,
      "brands": [],
      "constraints": [],
      "preferences": []
    }}

    Allowed price operators:
    "<=" | ">=" | "between" | "approx"

    Important rules:

    1. Do NOT invent brands.
    2. Extract only brands explicitly mentioned.
    3. If the user includes mathematical expressions (e.g. "radice di 60",
       "square root of 60", "60/2", "2*30"), evaluate them numerically.
    4. If a single numeric price appears, use operator:
       - "<=" for max/massimo/sotto/fino a
       - ">=" for minimo/almeno/sopra
       - "approx" for circa/intorno/sui
    5. semantic_query must contain ONLY meaningful search keywords.
       Remove price operators, numbers, mathematical expressions and filler words.
    6. If unsure about a field, set it to null or empty list.
    7. Output JSON only.

    Examples:

    Input: "cintura lacoste massimo 60 euro"
    Output:
    {{"semantic_query":"cintura Lacoste","product":"cintura","brands":["Lacoste"],"constraints":[{{"type":"price","operator":"<=","value":60}}],"preferences":[]}}

    Input: "cintura lacoste max radice di 60 euro"
    Output:
    {{"semantic_query":"cintura Lacoste","product":"cintura","brands":["Lacoste"],"constraints":[{{"type":"price","operator":"<=","value":7.75}}],"preferences":[]}}

    Query: {json.dumps(query, ensure_ascii=False)}
    """.strip()

    response, used_provider = call_llm(prompt, provider=provider)

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
# MERGE (LLM-first but deterministic price)
# ============================================================

def _get_price_constraints(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in items if c.get("type") == "price"]

def _get_condition_constraint(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((c for c in items if c.get("type") == "condition"), None)

def merge_results(rule_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    final = empty_result(rule_result["original_query"])

    # --------------------------------------------------
    # BRANDS
    # --------------------------------------------------
    llm_brands = llm_result.get("brands", []) if llm_result else []
    rule_brands = rule_result.get("brands", [])
    final["brands"] = filter_valid_brands(
        dedupe_keep_order(llm_brands + rule_brands)
    )

    # --------------------------------------------------
    # PRODUCT
    # --------------------------------------------------
    final["product"] = (
        (llm_result.get("product") if llm_result else None)
        or rule_result.get("product")
    )

    # --------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------
    rule_constraints = rule_result.get("constraints", [])
    llm_constraints = llm_result.get("constraints", []) if llm_result else []

    rule_price = _get_price_constraints(rule_constraints)
    llm_price = _get_price_constraints(llm_constraints)

    price_constraints = rule_price if rule_price else llm_price

    cond = (
        _get_condition_constraint(llm_constraints)
        or _get_condition_constraint(rule_constraints)
    )

    final["constraints"] = dedupe_keep_order(
        price_constraints + ([cond] if cond else [])
    )

    # --------------------------------------------------
    # PREFERENCES
    # --------------------------------------------------
    final["preferences"] = llm_result.get("preferences", []) if llm_result else []

    # --------------------------------------------------
    # CLEAN SEMANTIC QUERY
    # --------------------------------------------------
    semantic = (
        (llm_result.get("semantic_query") if llm_result else None)
        or rule_result.get("semantic_query")
        or rule_result["original_query"]
    )

    semantic = correct_brands_in_text(semantic)
    semantic = clean_semantic_query(semantic)

    # rimuovi eventuali parole isolate di 1 carattere
    semantic = re.sub(r"\b\w\b", "", semantic)
    semantic = re.sub(r"\s+", " ", semantic).strip()


    # fallback se vuoto
    if not semantic:
        parts = []
        if final["brands"]:
            parts.extend(final["brands"])
        if final["product"]:
            parts.append(final["product"])
        semantic = " ".join(parts) if parts else rule_result["original_query"]

    final["semantic_query"] = semantic.strip()

    return final

# ============================================================
# CONFIDENCE
# ============================================================

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

# ============================================================
# MAIN SERVICE
# ============================================================

def parse_query_service(
    text: str,
    llm_engine: Optional[str] = None,
    include_meta: bool = True
) -> Dict[str, Any]:

    original_query = normalize_text(text)
    working_text = correct_brands_in_text(original_query)

    # -----------------------------
    # Decide modalità LLM
    # -----------------------------
    if llm_engine == "rule_based":
        use_llm = False
        provider = None
    elif llm_engine in {"ollama", "gemini"}:
        use_llm = True
        provider = llm_engine
    else:
        # fallback a default config
        use_llm = True
        provider = None

    # -----------------------------
    # Rule-based sempre attivo
    # -----------------------------
    rule_result = rule_based_parse(working_text)

    llm_result = None
    used_provider = None

    if use_llm:
        llm_result, used_provider = llm_parse(working_text, provider=provider)

    final = merge_results(rule_result, llm_result)

    # ripristina query originale
    final["original_query"] = original_query

    if include_meta:
        final["_meta"] = {
            "llm_enabled": use_llm,
            "llm_success": llm_result is not None,
            "llm_provider": used_provider if use_llm else None,
            "confidence": compute_confidence(final, llm_result),
        }

    return final

# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    queries = [
        "vorrei un orologio garming usato intorno ai 200 euro",
        "vorrei delle scarpe abidas 100 euro max",
        "scarpe adidas intorno ai 100 euro",
        "iphone 13 massimo 1000 euro",
        "tra 200 e 400 euro cuffie bose",
        "cerco samsung usato sui 500 euro",
    ]
    for q in queries:
        parsed = parse_query_service(q, llm_engine="ollama", include_meta=True)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print("-" * 80)