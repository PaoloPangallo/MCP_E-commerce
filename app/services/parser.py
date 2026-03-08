
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
from typing import Any, Dict, List, Optional, Tuple, cast
from rapidfuzz import process, fuzz

# Gemini REST client
import requests
import spacy

# ============================================================
# LOAD .env (optional)
# ============================================================
try:
    from dotenv import load_dotenv  # pip install python-dotenv

    # tenta root progetto: .../app/services/parser.py -> parents[2] = root (di solito)
    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
    # fallback: cerca comunque .env nella cwd
    load_dotenv(override=False)
except Exception:
    # se python-dotenv non è installato, userà solo le env del sistema
    pass

# ============================================================
# CONFIG
# ============================================================

SPACY_MODEL = "it_core_news_sm"

# LLM provider: "ollama" | "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()  # es. "ollama"

# Ollama
OLLAMA_MODEL = os.getenv("PARSER_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:latest")).split('#')[0].strip()
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()  # <-- NON hardcodata: arriva da .env

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Lista piccola e opzionale: SOLO canonicalizzazione/fallback (non dipendere solo da questa)
BRAND_WHITELIST = {
    # Tech
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
    "hp": "HP",
    "dell": "Dell",
    "asus": "ASUS",
    "acer": "Acer",
    "lenovo": "Lenovo",
    "msi": "MSI",
    "lg": "LG",
    "dyson": "Dyson",
    "bose": "Bose",
    "jbl": "JBL",
    
    # Fashion & Luxury
    "ck": "Calvin Klein",
    "calvin": "Calvin Klein",
    "th": "Tommy Hilfiger",
    "tommy": "Tommy Hilfiger",
    "rl": "Ralph Lauren",
    "ralph": "Ralph Lauren",
    "lv": "Louis Vuitton",
    "vuitton": "Louis Vuitton",
    "dg": "Dolce & Gabbana",
    "dolce": "Dolce & Gabbana",
    "gabbana": "Dolce & Gabbana",
    "armani": "Giorgio Armani",
    "ea7": "Emporio Armani",
    "gucci": "Gucci",
    "prada": "Prada",
    "versace": "Versace",
    "nike": "Nike",
    "adidas": "Adidas",
    "puma": "Puma",
    "reebok": "Reebok",
    "levis": "Levi's",
    "levìs": "Levi's",
}
# Brand vocabulary dinamica (popolata dal layer search)
BRAND_VOCAB: List[str] = []


# ============================================================
# BRAND VOCAB LOAD (offline da brand_vocab.json)
# ============================================================

def load_brand_vocab() -> List[str]:
    try:
        root = Path(__file__).resolve().parents[2]  # MCP_ECOM root
        file_path = root / "brand_vocab.json"

        if not file_path.exists():
            logger.warning("brand_vocab.json not found at %s", file_path)
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            brands = json.load(f)

        if not isinstance(brands, list):
            logger.warning("brand_vocab.json invalid format")
            return []

        # normalizzazione pulita
        clean = []
        for b in brands:
            if isinstance(b, str):
                b = b.strip()
                if b:
                    clean.append(b)

        logger.info("Loaded %d brands into BRAND_VOCAB", len(clean))
        return sorted(set(clean))

    except Exception as e:
        logger.warning("Failed loading brand_vocab.json: %s", e)
        return []


BRAND_VOCAB: List[str] = load_brand_vocab()

CONDITION_SYNONYMS = {
    "new": {"nuovo", "nuova", "sigillato", "mai usato"},
    "used": {"usato", "usata", "seconda mano"},
    "refurbished": {"ricondizionato", "rigenerato", "refurbished"},
}

# Termini “vuoti”/vaghi: filtro anti-rumore (MINIMO)
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
    "compatibilities": {},  # dynamic properties for compatibility_filter
    "constraints": [],  # hard filters
    "preferences": [],  # soft wishes (ranking hints)
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


# ============================================================
# PRICE EXTRACTION
# ============================================================

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


def extract_base_price(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Restituisce (min_price, max_price)."""
    normalized = normalize_for_matching(text)
    
    # 🔥 PROTEGGIAMO "Air Max" e "taglia 43" per evitare "Max 270" -> Max Price 270 
    # o numeri che non sono prezzi
    normalized = re.sub(r"air\s+max", "AIR_MAX_MODEL", normalized)
    normalized = re.sub(r"(?:taglia|numero|misura)\s+\d+", "SIZE_OR_NUM", normalized)
    normalized = re.sub(r"(\d{1,2}|40|41|42|43|44|45|s|m|l|xl|xxl)\s*(?:[.,]\s*)?$", "SIZE_AT_END", normalized) # taglia alla fine

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
    # Filter: ignore 0 or near-zero prices as they are usually extraction errors for generic queries
    explicit_prices = [p for p in explicit_prices if p is not None and p > 0.1]
    if len(explicit_prices) == 1:
        return None, explicit_prices[0]

    return None, None


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
# BRAND EXTRACTION
# ============================================================


def fuzzy_brand_detection(text: str, threshold: int = 85) -> List[str]:
    if not BRAND_VOCAB:
        return []

    words = re.findall(r"\b\w+\b", text.lower())
    found = []

    for word in words:
        if len(word) < 4:
            continue

        match = process.extractOne(
            word,
            cast(List[str], BRAND_VOCAB),
            scorer=fuzz.partial_ratio
        )

        if match:
            brand, score, _ = match

            # Evita match troppo diversi in lunghezza
            if score >= threshold and abs(len(word) - len(brand)) <= 3:
                found.append(brand)

    return dedupe_keep_order(found)


def update_brand_vocab(brands: List[str]):
    global BRAND_VOCAB
    for b in brands:
        if isinstance(b, str) and b.strip():
            if b not in BRAND_VOCAB:
                BRAND_VOCAB.append(b)


def extract_brands(doc, original_text: str) -> List[str]:
    found: List[str] = []
    text_norm = normalize_for_matching(original_text)

    # 1️⃣ whitelist
    for raw, canonical in BRAND_WHITELIST.items():
        if re.search(rf"\b{re.escape(raw)}\b", text_norm):
            found.append(canonical)

    # 2️⃣ fuzzy matching su vocab reale
    fuzzy_found = fuzzy_brand_detection(original_text)
    found.extend(fuzzy_found)

    # 3️⃣ NER solo se nel vocabolario ufficiale
    for ent in getattr(doc, "ents", []):
        candidate = ent.text.strip(" ,.-")

        if not candidate:
            continue

        # 🔥 VALIDAZIONE FORTE
        if candidate in BRAND_VOCAB:
            found.append(candidate)

    return dedupe_keep_order(found)


# ============================================================
# PRODUCT EXTRACTION (fallback conservativo)
# ============================================================

def extract_product(doc, original_text: str, brands: List[str]) -> Optional[str]:
    """
    Fallback leggero e conservativo.
    NB: l'LLM è la fonte principale per product.
    """
    brand_norm = {b.lower() for b in brands}
    candidates: List[str] = []

    # noun_chunks
    try:
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if not text:
                continue
            text_norm = text.lower()

            if any(b in text_norm for b in brand_norm):
                continue
            if re.search(r"\d+\s*(euro|€)?", text_norm):
                continue

            forbidden = {"prezzo", "euro", "minimo", "massimo", "massima", "minima", "costo"}
            if any(f in text_norm.split() for f in forbidden):
                continue

            tokens = [t.text for t in chunk if not t.is_punct and not t.is_space]
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

    # fallback token-based
    content_tokens: List[str] = []
    forbidden_tokens = {"prezzo", "euro", "minimo", "massimo", "costo", "sotto", "sopra", "circa", "intorno", "entro"}
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.text.lower() in brand_norm or token.text.lower() in forbidden_tokens:
            continue
        if re.fullmatch(r"\d+(?:[.,]\d+)?", token.text):
            continue
        if token.pos_ in {"NOUN", "PROPN"}:
            content_tokens.append(token.text)

    if content_tokens:
        best = " ".join(content_tokens[:4]).strip()
        return None if is_vague_product(best) else best

    return None


# ============================================================
# RULE-BASED PARSER
# ============================================================

def rule_based_parse(query: str) -> Dict[str, Any]:
    normalized = normalize_text(query)
    doc = nlp(normalized)

    brands = extract_brands(doc, query)
    min_price, max_price = extract_base_price(query)
    condition = extract_condition(query)
    product = extract_product(doc, query, brands)

    result = empty_result(query)
    result["brands"] = brands
    result["product"] = product
    result["semantic_query"] = query  # raw per embeddings; pulizia la fa LLM

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
# LLM PROVIDERS
# ============================================================

def call_ollama(prompt: str) -> Optional[str]:
    try:
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 300
            }
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
    logger.info(f"[{request_id}] GEMINI call started")

    if not GEMINI_API_KEY:
        logger.error(f"[{request_id}] GEMINI_API_KEY missing")
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        t0 = time.time()

        response = requests.post(
            url,
            json=payload,
            timeout=GEMINI_TIMEOUT
        )

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[{request_id}] Gemini HTTP {response.status_code} in {elapsed}s")

        if response.status_code == 429:
            logger.warning(f"[{request_id}] Gemini rate limit (429)")
            return None

        if response.status_code != 200:
            logger.error(f"[{request_id}] Gemini error body: {response.text[:300]}")
            return None

        data = response.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )

        if not text:
            logger.warning(f"[{request_id}] Gemini returned empty text")
            return None

        logger.info(f"[{request_id}] Gemini success ({len(text)} chars)")
        return text.strip()

    except requests.Timeout:
        logger.error(f"[{request_id}] Gemini TIMEOUT after {GEMINI_TIMEOUT}s")
        return None

    except Exception as e:
        logger.exception(f"[{request_id}] Gemini unexpected error: {e}")
        return None


def call_llm(prompt: str) -> Tuple[Optional[str], str]:
    """
    Ritorna: (output, provider_usato)
    - Usa LLM_PROVIDER come principale
    - Se fallisce e LLM_FALLBACK_PROVIDER è impostato, prova fallback (es. ollama)
    """
    primary = LLM_PROVIDER
    fallback = LLM_FALLBACK_PROVIDER

    def _call(provider: str) -> Optional[str]:
        if provider == "ollama":
            return call_ollama(prompt)
        if provider == "gemini":
            return call_gemini(prompt)
        logger.error("Unknown provider: %s", provider)
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
                return text[start: i + 1]
    return None


# ============================================================
# VALIDATION / NORMALIZATION
# ============================================================

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


def clean_semantic_query(query: str) -> str:
    """Rimuove programmaticamente parole di disturbo dalla query di ricerca."""
    if not query:
        return ""
    
    # Parole da rimuovere (noise)
    forbidden = {
        "budget", "euro", "prezzo", "costo", "massimo", "minimo", "circa", "intorno",
        "taglia", "misura", "size", "color", "colore", "vorrei", "cerco", "trovami",
        "grazie", "ciao", "salve", "per", "come", "alla", "sotto", "sopra", "entro",
        "spedizione", "venditore", "nuovo", "usato", "condizione", "brand", "marca"
    }
    
    # Rimuoviamo simboli di valuta e punteggiatura
    query = re.sub(r"[€$£]", "", query)
    words = query.split()
    
    # Filtriamo le parole, mantenendo solo quelle non proibite e non numeriche pure (a meno che non siano taglie o modelli)
    clean_words = []
    for w in words:
        clean_w = w.lower().strip(" ,.!?()[]\"")
        if clean_w not in forbidden and len(clean_w) > 1:
            # Se è un numero puro e non sembra un anno o un modello tech, potremmo volerlo togliere, 
            # ma per ora siamo conservativi e togliamo solo il noise testuale.
            clean_words.append(w)
            
    return " ".join(clean_words).strip()

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
    if isinstance(semantic_query, str) and semantic_query.strip():
        result["semantic_query"] = clean_semantic_query(semantic_query.strip())
    else:
        result["semantic_query"] = clean_semantic_query(original_query)

    brands = data.get("brands", [])
    if isinstance(brands, list):
        normalized_brands: List[str] = []
        for b in brands:
            if isinstance(b, str) and b.strip():
                normalized_brands.append(normalize_brand(b))
        result["brands"] = dedupe_keep_order(normalized_brands)

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
            # Safeguard: No 0 or near-0 price constraints allowed (hallucination or wrong extraction)
            if nc and nc.get("type") == "price":
                val = nc.get("value")
                if nc.get("operator") == "between":
                   if isinstance(val, list) and max(val) <= 0.1: continue
                elif isinstance(val, (int, float)) and val <= 0.1:
                    continue
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

    compatibilities = data.get("compatibilities", {})
    if isinstance(compatibilities, dict):
        result["compatibilities"] = {str(k): str(v) for k, v in compatibilities.items() if
                                     str(k).strip() and str(v).strip()}

    return result


def enforce_numeric_consistency(original_query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evita che l'LLM cambi numeri del modello (es 15 -> 13).
    Se il numero nel product non compare nella query originale,
    il product viene scartato.
    """
    import re

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

def llm_parse(query: str, previous_context: Dict[str, Any] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    context_part = ""
    if previous_context:
        ctx_clean = {k: v for k, v in previous_context.items() if not k.startswith("_")}
        context_part = f"\nPREVIOUS SHOPPING CONTEXT: {json.dumps(ctx_clean, ensure_ascii=False)}\nUSE CONTEXT TO: Resolve pronouns (li, quello), maintain core product/brand if user only adds attributes (color, size), and detect topic changes.\n"

    prompt = f"""
You are a smart shopping assistant state manager. Your goal is to extract entities and maintain the current search context.

INPUTS:
- CURRENT REQUEST: {json.dumps(query, ensure_ascii=False)}
- PREVIOUS CONTEXT: {json.dumps(ctx_clean, ensure_ascii=False) if previous_context else "None"}

STATE TRANSITION LOGIC:
1. TOPIC REDIRECTION (CRITICAL): If the user mentions a BRAND (e.g. "Nike") or a PRODUCT (e.g. "scarpe") that is DIFFERENT from the PREVIOUS CONTEXT, you must IMMEDIATELY DISCARD all old brand/product information. DO NOT MERGE multiple brands.
2. ATTRIBUTE OVERRIDE: If the user provides a NEW value for an attribute (e.g., a new budget "80 euro" vs old "60", or a new size L vs old M), the NEW value must ALWAYS replace the old one. NEVER keep two different values for the same attribute.
3. CONTEXT PRESERVATION: Only keep context (like size or color) if the user is refining the SAME brand/product AND doesn't provide a new value for that attribute.
4. SEMANTIC QUERY: Generate a keyword-rich string for eBay. Use ONLY: [Brand] [Product] [Size] [Color] [Gender]. 
   - DO NOT include "budget", "euro", "taglia", "prezzo", "vorrei", "cerco".
   - DO NOT include technical field names or JSON keys.
   - Example: "Calvin Klein felpa L nero uomo"
5. CLEANING: Remove ALL conversational noise and filler words.

OUTPUT FORMAT (JSON ONLY):
{{
  "semantic_query": "", 
  "product": "",      
  "brands": [],
  "compatibilities": {{}}, 
  "constraints": [],
  "preferences": [],
  "missing_info": []    
}}

RULES:
- Standardize sizes (EU 44, L, XL, W32 L30).
- Standardize genders (Men, Women, Kids) in `compatibilities`.
- NO PRICE IN QUERY: Never include price, budget, or currency (e.g., "60 euro", "budget", "€") in the `semantic_query`. Put them ONLY in `constraints`.
- If the user says "red or blue", prioritize the first or include both if relevant.
- NEVER combine two competing brands in one `semantic_query` unless the user explicitly asks for both.
"""

    response, used_provider = call_llm(prompt)
    if not response:
        return None, used_provider

    json_text = extract_first_json_object(response)
    if not json_text:
        logger.warning("No JSON object found in LLM output")
        return None, used_provider

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from LLM: %s", response[:200])
        return None, used_provider

    if not isinstance(raw, dict):
        logger.warning("LLM output is not a JSON object")
        return None, used_provider

    parsed = validate_llm_result(raw, query)
    parsed = enforce_numeric_consistency(query, parsed)
    return parsed, used_provider


# ============================================================
# MERGE (LLM-first)
# ============================================================

def merge_results(rule_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Se LLM fallisce → puro rule-based
    if not llm_result:
        # costruiamo semantic_query pulita
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

    # ------------------------------------------------------------
    # 1️⃣ semantic_query → LLM
    # ------------------------------------------------------------
    final["semantic_query"] = (
            llm_result.get("semantic_query")
            or rule_result.get("semantic_query")
            or rule_result["original_query"]
    )

    # ------------------------------------------------------------
    # 2️⃣ PRODUCT → LLM SOLO SE CONSISTENTE
    # ------------------------------------------------------------
    final["product"] = llm_result.get("product")

    # ------------------------------------------------------------
    # 3️⃣ BRANDS → unione ma pulita
    # ------------------------------------------------------------
    llm_brands = llm_result.get("brands", []) or []
    rule_brands = rule_result.get("brands", []) or []

    final["brands"] = dedupe_keep_order(llm_brands + rule_brands)

    # ------------------------------------------------------------
    # 4️⃣ CONSTRAINTS
    #    Preferisci LLM sia per PREZZO che per CONDIZIONE, fallback su rule-based
    # ------------------------------------------------------------

    rule_constraints = rule_result.get("constraints", [])
    llm_constraints = llm_result.get("constraints", [])

    # --- PRICE
    llm_price_constraints = [c for c in llm_constraints if c.get("type") == "price"]
    price_constraints = llm_price_constraints if llm_price_constraints else [c for c in rule_constraints if
                                                                             c.get("type") == "price"]

    # --- CONDITION
    llm_condition = next(
        (c for c in llm_constraints if c.get("type") == "condition"),
        None
    )

    if llm_condition:
        condition_constraints = [llm_condition]
    else:
        condition_constraints = [
            c for c in rule_constraints if c.get("type") == "condition"
        ]

    final["constraints"] = price_constraints + condition_constraints

    # ------------------------------------------------------------
    # 5️⃣ PREFERENCES → solo LLM
    # ------------------------------------------------------------
    final["preferences"] = llm_result.get("preferences", []) or []

    # ------------------------------------------------------------
    # 6️⃣ COMPATIBILITÀ → solo LLM
    # ------------------------------------------------------------
    final["compatibilities"] = llm_result.get("compatibilities", {}) or {}

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

def correct_brands_in_text(text: str) -> str:
    words = text.split()
    corrected = []

    for w in words:
        w_low = w.lower().strip(",.!?")
        
        # PRIORITÀ 1: Whitelist (anche se corto, es. 'ck')
        if w_low in BRAND_WHITELIST:
            corrected.append(BRAND_WHITELIST[w_low])
            continue
            
        # PRIORITÀ 2: Fuzzy matching su vocabolo dinamico
        if not BRAND_VOCAB or len(w) < 4:
            corrected.append(w)
            continue

        match = process.extractOne(
            w,
            cast(List[str], BRAND_VOCAB),
            scorer=fuzz.partial_ratio
        )

        if match:
            brand, score, _ = match
            if score >= 88 and abs(len(w) - len(brand)) <= 3:
                corrected.append(brand)
                continue

        corrected.append(w)

    return " ".join(corrected)


def parse_query_service(text: str, use_llm: bool = True, include_meta: bool = True, previous_context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    text = normalize_text(text)
    if previous_context:
        logger.info(f"PARSER: Using previous context for query: '{text}'")

    # 🔥 spell correction prima di tutto
    text = correct_brands_in_text(text)

    rule_result = rule_based_parse(text)

    llm_result = None
    used_provider: Optional[str] = None

    if use_llm:
        parsed_llm, used_provider = llm_parse(text, previous_context)
        llm_result = parsed_llm

    final = merge_results(rule_result, llm_result)

    if include_meta:
        final["_meta"] = {
            "llm_enabled": use_llm,
            "llm_success": llm_result is not None,
            "llm_provider": used_provider if use_llm else None,
            "confidence": compute_confidence(final, llm_result),
        }
    else:
        final.pop("_meta", None)

    return final


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    queries = [
        "Cerco un iPhone 14 usato sotto 600 euro",
        "Vorrei una PlayStation 5 nuova entro 450 euro",
        "Laptop Lenovo o ASUS tra 500 e 800 euro",
        "Mi serve un Dyson ricondizionato",
        "Cerco cuffie Bose o JBL, meglio nuove ma accetto usate",
        "Avvisami se un iPhone 14 scende sotto 500 euro",
        "Meglio Samsung o Xiaomi sui 500 euro?",
        "Secondo te 650 euro per questo iPhone 14 usato conviene?",
        "vorrei qualcosa tipo samsung o xiaomi non troppo caro diciamo sui 500, se nuovo anche 600, fammi sapere quando esce qualcosa di nuovo",
        "sono interessato a comprare un macbook air m3 massimo 800 da 1 terabyte, vecchio massimo un anno",
    ]

    for q in queries:
        parsed = parse_query_service(q, use_llm=True, include_meta=True)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print("-" * 80)