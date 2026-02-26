"""
app/services/parser.py

Parser NLP "neutro" e coerente con quanto deciso:
- NESSUNA logica di business / intent (te la gestisci tu fuori)
- Output stabile: semantic_query, product, brands, constraints, preferences, _meta
- Parsing minimalista ma utile per retrieval: brand/product/price/condition
- LLM come componente principale + fallback rule-based
- Validazione/normalizzazione robusta
"""

import json
import logging
import re
import subprocess
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import spacy

# ============================================================
# CONFIG
# ============================================================

SPACY_MODEL = "it_core_news_sm"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 60

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Lista piccola e opzionale: SOLO canonicalizzazione/fallback (non dipendere solo da questa)
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

# Termini “vuoti”/vaghi: filtro anti-rumore (MINIMO, non “mille termini”)
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
    "constraints": [],   # hard filters
    "preferences": [],   # soft wishes (ranking hints)
    "_meta": {
        "llm_used": False,
        "llm_success": False,
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
    # fallback: semantic_query = query originale (utile per embeddings/FAISS)
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
    # fallback semplice
    return raw[0].upper() + raw[1:] if len(raw) > 1 else raw.upper()

def normalize_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        s = value.strip().lower()
        s = s.replace("€", "").replace("euro", "").strip()

        # "1.200,50" -> 1200.50
        # "1.200" -> 1200
        # "299,99" -> 299.99
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
    # “qualcosa …” / “una cosa …” ecc
    if t.startswith("qualcosa"):
        return True
    return False

# ============================================================
# PRICE EXTRACTION
# ============================================================

NUM_PATTERN = r"(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)"

PRICE_RANGE_PATTERNS = [
    re.compile(
        rf"\b(?:tra|da)\s+{NUM_PATTERN}\s*(?:euro|€)?\s+(?:e|a)\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
]

MAX_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:sotto|meno di|massimo|max|fino a|entro|non oltre)\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
    re.compile(
        rf"\b(?:circa|intorno a|sui|sulle|verso)\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
        re.IGNORECASE
    ),
]

MIN_PRICE_PATTERNS = [
    re.compile(
        rf"\b(?:almeno|minimo|min|sopra|oltre|più di)\s+{NUM_PATTERN}\s*(?:euro|€)?\b",
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

    for pattern in PRICE_RANGE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            p1 = normalize_float(m.group(1))
            p2 = normalize_float(m.group(2))
            if p1 is not None and p2 is not None:
                return min(p1, p2), max(p1, p2)

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

def extract_brands(doc, original_text: str) -> List[str]:
    found: List[str] = []
    text_norm = normalize_for_matching(original_text)

    # 1) Canonicalizzazione da whitelist (substring word-boundary)
    for raw, canonical in BRAND_WHITELIST.items():
        if re.search(rf"\b{re.escape(raw)}\b", text_norm):
            found.append(canonical)

    # 2) Named entities ORG/PRODUCT
    for ent in getattr(doc, "ents", []):
        if ent.label_ in {"ORG", "PRODUCT"}:
            candidate = ent.text.strip(" ,.-")
            if candidate:
                found.append(normalize_brand(candidate))

    # 3) Fallback: sequenze PROPN
    current: List[str] = []
    for token in doc:
        if token.pos_ == "PROPN":
            current.append(token.text)
        else:
            if current:
                found.append(normalize_brand(" ".join(current)))
                current = []
    if current:
        found.append(normalize_brand(" ".join(current)))

    cleaned = []
    for b in found:
        b = (b or "").strip(" ,.-")
        if b:
            cleaned.append(b)

    # Dedup case-insensitive
    return dedupe_keep_order(cleaned)

# ============================================================
# PRODUCT EXTRACTION (fallback conservativo)
# ============================================================

def extract_product(doc, original_text: str, brands: List[str]) -> Optional[str]:
    """
    Fallback leggero e conservativo.
    NB: l'LLM è la fonte principale per product. Qui cerchiamo un backup sensato.
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
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.text.lower() in brand_norm:
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
    result["semantic_query"] = query  # tieni sempre raw per embeddings; pulizia la fa LLM

    # Hard constraints
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
# LLM CALL
# ============================================================

def call_llm(prompt: str, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        if proc.returncode != 0:
            logger.warning("LLM returned non-zero exit code: %s", proc.returncode)
            if proc.stderr:
                logger.warning("LLM stderr: %s", proc.stderr.strip())
            return None

        out = (proc.stdout or "").strip()
        if not out:
            logger.warning("LLM returned empty output")
            return None
        return out

    except subprocess.TimeoutExpired:
        logger.warning("LLM timeout after %s seconds", timeout)
        return None
    except FileNotFoundError:
        logger.warning("Ollama non trovato nel PATH")
        return None
    except Exception as e:
        logger.exception("LLM error: %s", e)
        return None

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

    # semantic_query
    semantic_query = data.get("semantic_query")
    result["semantic_query"] = (
        semantic_query.strip()
        if isinstance(semantic_query, str) and semantic_query.strip()
        else original_query
    )

    # brands
    brands = data.get("brands", [])
    if isinstance(brands, list):
        normalized_brands: List[str] = []
        for b in brands:
            if isinstance(b, str) and b.strip():
                normalized_brands.append(normalize_brand(b))
        result["brands"] = dedupe_keep_order(normalized_brands)

    # product
    product = data.get("product")
    if isinstance(product, str) and product.strip().lower() not in {"", "null", "none"}:
        cleaned = product.strip()
        result["product"] = None if is_vague_product(cleaned) else cleaned
    else:
        result["product"] = None

    # constraints
    constraints = data.get("constraints", [])
    if isinstance(constraints, list):
        norm_constraints: List[Dict[str, Any]] = []
        for c in constraints:
            nc = normalize_constraint(c)
            if nc is not None:
                norm_constraints.append(nc)
        result["constraints"] = dedupe_keep_order(norm_constraints)

    # preferences
    preferences = data.get("preferences", [])
    if isinstance(preferences, list):
        norm_prefs: List[Dict[str, Any]] = []
        for p in preferences:
            np = normalize_preference(p)
            if np is not None:
                norm_prefs.append(np)
        result["preferences"] = dedupe_keep_order(norm_prefs)

    return result

# ============================================================
# LLM PARSE
# ============================================================

def llm_parse(query: str) -> Optional[Dict[str, Any]]:
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
  "constraints": [],
  "preferences": []
}}

Definitions:
- semantic_query: short cleaned version of the query useful for semantic retrieval.
- product: the main item/model explicitly mentioned (e.g. "iPhone 15", "PS5", "MacBook Air M3"). Use null only if no item/model is stated.
- brands: brands explicitly mentioned.
- constraints: hard filters that restrict search results.
- preferences: soft wishes that influence ranking but should NOT exclude results.

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
- If the user gives a MAX BUDGET (e.g. "massimo 1000 euro", "entro 500"), use ONLY "<=" (not both >= and <=).
- If the user gives a MINIMUM budget (e.g. "almeno 300 euro"), use ONLY ">=".
- Use "between" only when a range is explicitly given.
- Output JSON only.

Examples (must follow exactly):
Query: "iphone 13 massimo 1000 euro"
Output: {{"semantic_query":"iphone 13","product":"iPhone 13","brands":["Apple"],"constraints":[{{"type":"price","operator":"<=","value":1000}}],"preferences":[]}}

Query: "tra 200 e 400 euro cuffie bose"
Output: {{"semantic_query":"cuffie bose","product":"cuffie","brands":["Bose"],"constraints":[{{"type":"price","operator":"between","value":[200,400]}}],"preferences":[]}}

Query: {json.dumps(query, ensure_ascii=False)}
""".strip()

    response = call_llm(prompt)
    if not response:
        return None

    json_text = extract_first_json_object(response)
    if not json_text:
        logger.warning("No JSON object found in LLM output")
        return None

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from LLM: %s", response[:200])
        return None

    if not isinstance(raw, dict):
        logger.warning("LLM output is not a JSON object")
        return None

    return validate_llm_result(raw, query)
# ============================================================
# MERGE
# ============================================================

def merge_results(rule_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Se LLM fallisce -> puro rule-based
    if not llm_result:
        return rule_result

    # Se LLM riesce -> LLM è la fonte principale.
    # Rule-based usato SOLO per "riempire buchi" in modo conservativo.
    final = empty_result(rule_result["original_query"])

    # semantic_query: preferisci LLM, fallback su rule, fallback su originale
    final["semantic_query"] = (
        llm_result.get("semantic_query")
        or rule_result.get("semantic_query")
        or rule_result["original_query"]
    )

    # product: se LLM ha risposto, fidati ANCHE se è null.
    # Questo evita che il rule-based inietti robaccia tipo "massimo euro".
    final["product"] = llm_result.get("product")

    # brands: preferisci LLM; se vuote, puoi unire col rule-based (utile quando LLM manca qualche brand)
    llm_brands = llm_result.get("brands", []) or []
    rule_brands = rule_result.get("brands", []) or []
    final["brands"] = dedupe_keep_order(llm_brands + rule_brands)

    # constraints: preferisci LLM; se vuote, fallback su rule-based
    llm_constraints = llm_result.get("constraints", [])
    final["constraints"] = llm_constraints if llm_constraints else (rule_result.get("constraints", []) or [])

    # preferences: LLM
    final["preferences"] = llm_result.get("preferences", []) or []

    return final

# ============================================================
# CONFIDENCE
# ============================================================

def compute_confidence(final_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> float:
    """
    Euristica semplice e onesta (non deve mai arrivare a 1.0).
    Non deve guidare decisioni critiche, solo UX/log.
    """
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

def parse_query_service(text: str, use_llm: bool = True, include_meta: bool = True) -> Dict[str, Any]:
    rule_result = rule_based_parse(text)

    llm_result = None
    if use_llm:
        llm_result = llm_parse(text)

    final = merge_results(rule_result, llm_result)

    if include_meta:
        final["_meta"] = {
            # "enabled" = hai provato ad usare l'LLM
            "llm_enabled": use_llm,
            # "success" = hai ottenuto JSON valido e normalizzato
            "llm_success": llm_result is not None,
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