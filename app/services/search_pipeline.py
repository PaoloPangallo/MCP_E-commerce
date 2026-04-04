import asyncio
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.redis import redis_client
from app.config.cache import SEARCH_PIPELINE_TTL
from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.feedback import get_seller_feedback
from app.services.metrics.ir_metrics import ndcg_at_k, precision_at_k, recall_at_k
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.parser import parse_query_service
from app.services.rag.context_builder import build_context
from app.services.rag.explainer import explain_results
from app.services.rag.retriever import retrieve_context
from app.services.rag.reranker import rerank_products
from app.services.rag.query_expansion import expand_query
from app.services.trust import compute_trust_score
from app.services.user_profiling import update_user_profile, get_dominant_condition
from app.llm.client import call_llm
from app.services.ebay_metadata import get_return_policies
from app.services.nlp_ner import extract_attributes_batch
from app.services.rag.qdrant_store import index_search_items

logger = logging.getLogger(__name__)

_CACHE_TTL = SEARCH_PIPELINE_TTL

MAX_RESULTS_FROM_EBAY = 15
MAX_SELLERS_FOR_TRUST = 5
MAX_FEEDBACK_PER_SELLER = 40
FEEDBACK_WORKERS = 6


def _normalize_llm_engine(llm_engine: str) -> str:
    llm_engine = (llm_engine or "").strip().lower()
    if llm_engine in {"gemini", "ollama", "rule_based"}:
        return llm_engine
    return "ollama"


def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids: Set[str] = set()
    deduped: List[Dict[str, Any]] = []

    for item in items:
        ebay_id = item.get("ebay_id")
        if ebay_id:
            if ebay_id in seen_ids:
                continue
            seen_ids.add(ebay_id)
        
        deduped.append(item)

    return deduped


def _build_ebay_query(parsed: Dict[str, Any], fallback_query: str) -> str:
    parts: List[str] = []
    
    brands = parsed.get("brands") or []
    product = parsed.get("product")
    
    if brands:
        parts.extend(str(b).strip() for b in brands if str(b).strip())
    if product:
        parts.append(str(product).strip())
        
    # Includiamo constraints testuali (no price/condition)
    constraints = parsed.get("constraints") or []
    for c in constraints:
        ctype = c.get("type")
        val = c.get("value")
        if ctype not in ("price", "condition", "aspect") and val:
            # Gestione negazioni per query eBay
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                v_str = str(v).strip()
                # Se la query originale contiene "no", "senza", "nè" riferiti a questo valore
                # cerchiamo di usare l'operatore NOT di eBay (-)
                neg_patterns = [r"\bno\b", r"\bsenza\b", r"\bn[èe]\b"]
                orig_low = fallback_query.lower()
                is_negated = any(re.search(pf + r"\s+" + re.escape(v_str.lower()), orig_low) for pf in neg_patterns)
                
                if is_negated:
                    parts.append(f"-{v_str}")
                else:
                    parts.append(v_str)
                
    if not parts:
        return fallback_query
        
    # Deduplicazione parole mantenendo ordine
    final_tokens: List[str] = []
    seen_tokens_low = set()
    
    raw_query = " ".join(parts)
    # CRITICAL DEBUG: Vediamo cosa stiamo mandando a eBay
    logger.debug("EBAY SEARCH QUERY: '%s' (Original fallback: '%s')", raw_query, fallback_query)
    logger.info("EBAY QUERY: %s", raw_query)
    
    for word in raw_query.split():
        w_low = word.lower()
        if w_low not in seen_tokens_low:
            final_tokens.append(word)
            seen_tokens_low.add(w_low)
            
    return " ".join(final_tokens).strip()


def _sanitize_floats(obj: Any) -> Any:
    """Ricorsivamente converte numpy floats o NaN/Inf in tipi Python-native."""
    import math
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(i) for i in obj]
    elif np and isinstance(obj, (np.integer,)):
        return int(obj)
    elif np and isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')): # NumPy scalars
        return obj.item()
    return obj


async def _fetch_feedback_cached(seller_name: str, limit: int = MAX_FEEDBACK_PER_SELLER) -> List[Dict[str, Any]]:
    seller_key = seller_name.strip().lower()
    cache_key = f"seller_feedback:{seller_key}"
    
    cached = redis_client.get_json(cache_key)
    if cached is not None:
        return cached

    # get_seller_feedback is now async
    feedbacks = await get_seller_feedback(seller_name, limit=limit) or []
    redis_client.set_json(cache_key, feedbacks, ttl_seconds=int(_CACHE_TTL))
    return feedbacks


async def _compute_seller_trust_cached(seller_name: str) -> Optional[float]:
    seller_key = seller_name.strip().lower()
    cache_key = f"seller_trust:{seller_key}"
    
    feedbacks = await _fetch_feedback_cached(seller_name, limit=MAX_FEEDBACK_PER_SELLER)
    if not feedbacks:
        return None

    cached = redis_client.get_json(cache_key)
    if cached is not None:
        if int(cached.get("count", -1)) == len(feedbacks):
            return int(float(cached["trust_score"]) * 1000) / 1000.0

    # These are CPU bound, but let's keep them in thread for now if complex, 
    # or just run them here if they are fast enough.
    sentiment_score = await asyncio.to_thread(compute_sentiment_score, feedbacks, max_texts=20)
    trust_score = await asyncio.to_thread(compute_trust_score, feedbacks, sentiment_score=sentiment_score)

    result = {
        "count": float(len(feedbacks)),
        "sentiment_score": float(sentiment_score),
        "trust_score": float(trust_score),
    }
    redis_client.set_json(cache_key, result, ttl_seconds=int(_CACHE_TTL))

    return int(float(trust_score) * 1000) / 1000.0


async def _prefetch_top_sellers_feedback(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute trust only for the most relevant sellers using async gather.
    """
    sellers: List[str] = []
    seen = set()

    for item in items:
        seller = item.get("seller_name")
        if seller and seller not in seen:
            seen.add(seller)
            sellers.append(seller)

        if len(sellers) >= MAX_SELLERS_FOR_TRUST:
            break

    scores: Dict[str, float] = {}
    if not sellers:
        return scores

    # Parallel async execution
    tasks = [_compute_seller_trust_cached(s) for s in sellers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for seller, score in zip(sellers, results):
        if isinstance(score, (float, int)):
            scores[seller] = score
        elif isinstance(score, Exception):
            logger.warning("Trust computation failed for seller=%s: %s", seller, score)

    return scores


def _batch_fetch_existing_ebay_ids(db: Session, items: List[Dict[str, Any]]) -> Set[str]:
    ebay_ids = [item.get("ebay_id") for item in items if item.get("ebay_id")]
    if not ebay_ids:
        return set()

    # Step A: Check Redis first
    existing_in_cache = set()
    to_query_db = []
    
    for eid in ebay_ids:
        if redis_client.set_is_member("known_ebay_ids", str(eid)):
            existing_in_cache.add(str(eid))
        else:
            to_query_db.append(eid)
            
    if not to_query_db:
        return existing_in_cache

    # Step B: Query DB for those not in cache
    rows = (
        db.query(Listing.ebay_id)
        .filter(Listing.ebay_id.in_(to_query_db))
        .all()
    )
    
    found_in_db = {str(row[0]) for row in rows}
    
    # Step C: Update cache with what we found in DB
    for eid in found_in_db:
        redis_client.set_add("known_ebay_ids", eid)
        
    return existing_in_cache.union(found_in_db)


def _prepare_and_persist_items(
    db: Session,
    items: List[Dict[str, Any]],
    seller_trust_map: Dict[str, float],
) -> tuple[List[Dict[str, Any]], int]:
    existing_ids = _batch_fetch_existing_ebay_ids(db, items)
    saved_count = 0
    results_out: List[Dict[str, Any]] = []

    logger.info("_prepare_and_persist_items: received %d items", len(items))

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            logger.warning("Skipping item without ebay_id: title=%s", item.get("title", "?"))
            continue

        already = ebay_id in existing_ids

        if not already:
            listing = Listing(
                ebay_id=ebay_id,
                title=item.get("title"),
                price=item.get("price"),
                currency=item.get("currency"),
                condition=item.get("condition"),
                seller_name=item.get("seller_name"),
                seller_rating=item.get("seller_rating"),
                url=item.get("url"),
                image_url=item.get("image_url"),
            )
            db.add(listing)
            saved_count += 1
            # Update cache so we don't query DB next time
            redis_client.set_add("known_ebay_ids", str(ebay_id))

        seller_name = item.get("seller_name")
        trust_score = seller_trust_map.get(seller_name) if seller_name else None

        item_copy = dict(item)
        item_copy["_already_in_db"] = already
        item_copy["trust_score"] = trust_score
        results_out.append(item_copy)

    return results_out, saved_count


def _apply_final_ranking(
    items: List[Dict[str, Any]],
    user: Optional[object] = None,
) -> None:
    for item in items:
        # Use the highly calibrated _final_score from the RAG reranker and Cross-Encoder
        base_relevance = float(item.get("_final_score", item.get("_rerank_score", 0)) or 0)
        trust = float(item.get("trust_score") or 0)
        price = item.get("price") or 0

        # Compute value_score (Value-for-money heuristic)
        condition_str = (item.get("condition") or "").lower()
        condition_score = 0.5
        if any(w in condition_str for w in ["nuovo", "new"]):
            condition_score = 1.0
        elif any(w in condition_str for w in ["ricondizionato", "refurbished"]):
            condition_score = 0.7
        elif any(w in condition_str for w in ["usato", "used", "gebraucht"]):
            condition_score = 0.4
            
        val_trust = trust if trust > 0 else 0.5  # Use average trust if not computed yet
        quality = 0.6 * val_trust + 0.4 * condition_score
        price_val = float(price)
        if price_val > 0:
            # Diminishing returns on price: cheaper is better
            price_factor = 1.0 / (1.0 + math.log1p(price_val / 100))
            # Normalise roughly to 0-1 range
            item["value_score"] = round(min(1.0, quality * (0.5 + price_factor)), 4)
        else:
            item["value_score"] = 0.0

        ranking_score = base_relevance

        explanations = []

        if user:
            favorite_brands = getattr(user, "favorite_brands", None)
            if favorite_brands:
                brands_pref = {
                    b.strip().lower()
                    for b in favorite_brands.split(",")
                    if b.strip()
                }

                title = (item.get("title") or "").lower()
                for b in brands_pref:
                    if b in title:
                        ranking_score += 0.15  # Increased from 0.10
                        item["brand_match"] = True
                        explanations.append(f"This item matches your preferred brand '{b}'.")
                        break

            # ── Condition preference bonus ──────────────────────────────
            try:
                dominant_condition = get_dominant_condition(user)
                if dominant_condition:
                    item_cond = (item.get("condition") or "").lower()
                    cond_match = (
                        (dominant_condition == "new" and any(w in item_cond for w in ["nuovo", "new"]))
                        or (dominant_condition == "used" and any(w in item_cond for w in ["usato", "used", "gebraucht"]))
                        or (dominant_condition == "refurbished" and any(w in item_cond for w in ["ricondizionato", "refurbished"]))
                    )
                    if cond_match:
                        ranking_score += 0.08
                        item["condition_match"] = True
                        explanations.append(f"Matches your usual condition preference ({dominant_condition}).")
            except Exception:
                pass

            # ── Granular Budget Matching ────────────────────────────────
            contextual_budgets = {}
            if user and getattr(user, "contextual_budgets", None):
                try:
                    contextual_budgets = json.loads(user.contextual_budgets)
                except Exception:
                    pass

            # Prefer contextual budget (auto_ or manual) over global
            best_pref = None
            title_lower = (item.get("title") or "").lower()

            # 1. Manual brand key (brand:nike → set via MCP)
            for b_key, b_val in contextual_budgets.items():
                if b_key.startswith("brand:") and b_key.split(":", 1)[1] in title_lower:
                    best_pref = float(b_val)
                    break

            # 2. Auto brand key (auto_brand:nike)
            if best_pref is None:
                for b_key, b_val in contextual_budgets.items():
                    if b_key.startswith("auto_brand:") and b_key.split(":", 1)[1] in title_lower:
                        best_pref = float(b_val)
                        break

            # 3. Auto category key (auto_cat:cellulari)
            if best_pref is None:
                item_cat = (item.get("category_name") or "").lower()
                for b_key, b_val in contextual_budgets.items():
                    if b_key.startswith("auto_cat:") and b_key.split(":", 1)[1] in item_cat:
                        best_pref = float(b_val)
                        break

            # 4. Legacy manual product key (product:main)
            if best_pref is None:
                intent_product = item.get("intent_product_type") or ""
                cat_key = f"product:{intent_product.lower()}"
                if cat_key in contextual_budgets:
                    best_pref = float(contextual_budgets[cat_key])

            # 5. Fallback to global price_preference
            if best_pref is None:
                price_pref = getattr(user, "price_preference", None)
                if price_pref:
                    try:
                        best_pref = float(price_pref)
                    except Exception:
                        pass

            if best_pref and price:
                try:
                    if float(price) <= best_pref:
                        ranking_score += 0.10  # Increased from 0.08
                        item["price_match"] = True
                        explanations.append(f"Matching your budget profile for this category (~{int(best_pref)}€).")
                except Exception:
                    pass

        if trust >= 0.8:
            explanations.append("Seller has very strong feedback and trust score.")
        elif trust >= 0.6:
            explanations.append("Seller shows generally positive feedback.")

        item["ranking_score"] = round(min(1.0, float(ranking_score)), 4)
        if explanations:
            item["explanations"] = explanations

    # Sort one final time to respect user preferences (like brand match bonuses) injected above
    items.sort(key=lambda x: x.get("ranking_score", 0), reverse=True)


async def _compute_llm_value_scores(
    top_items: List[Dict[str, Any]],
    all_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Usa l'LLM (Ollama) per valutare il 'Value for Money' dei top prodotti.
    Passa le statistiche di mercato calcolate dall'intera SERP eBay corrente
    come contesto — nessuna chiamata API esterna aggiuntiva.
    """
    if not top_items:
        return {}

    # ── Calcola statistiche di mercato dai risultati eBay completi ──────────
    prices = [float(it["price"]) for it in all_items if isinstance(it.get("price"), (int, float)) and it["price"] > 0]
    
    market_ctx: Dict[str, Any] = {}
    if prices:
        market_ctx["count"] = len(prices)
        market_ctx["avg_price"]    = round(sum(prices) / len(prices), 2)
        market_ctx["min_price"]    = round(min(prices), 2)
        market_ctx["max_price"]    = round(max(prices), 2)
        sorted_p = sorted(prices)
        mid = len(sorted_p) // 2
        market_ctx["median_price"] = round(
            sorted_p[mid] if len(sorted_p) % 2 else (sorted_p[mid - 1] + sorted_p[mid]) / 2, 2
        )

    # Distribuzione condizioni per contestualizzare l'LLM
    from collections import Counter
    cond_counter = Counter(
        (it.get("condition") or "unknown").lower()
        for it in all_items
    )
    market_ctx["condition_distribution"] = dict(cond_counter.most_common(5))

    # ── Prepara i candidati da valutare ─────────────────────────────────────
    candidates = []
    for item in top_items:
        price = item.get("price")
        price_vs_avg = None
        if prices and isinstance(price, (int, float)):
            avg = market_ctx["avg_price"]
            price_vs_avg = f"{round((price - avg) / avg * 100, 1):+.1f}% rispetto alla media"
        
        candidates.append({
            "id":            item.get("ebay_id"),
            "title":         (item.get("title") or "")[:70],
            "price":         f"{price} {item.get('currency', 'EUR')}",
            "price_vs_avg":  price_vs_avg,
            "condition":     item.get("condition"),
            "seller_trust":  item.get("trust_score"),
            "seller_rating": item.get("seller_rating"),
        })

    prompt = f"""Sei un esperto di mercato eBay. Valuta il rapporto qualità-prezzo di questi annunci.

CONTESTO MERCATO (prezzi reali dalla ricerca corrente):
- Prodotti trovati: {market_ctx.get('count', '?')}
- Prezzo medio: {market_ctx.get('avg_price', '?')} EUR
- Range prezzi: {market_ctx.get('min_price', '?')} – {market_ctx.get('max_price', '?')} EUR
- Prezzo mediano: {market_ctx.get('median_price', '?')} EUR
- Condizioni presenti: {market_ctx.get('condition_distribution', {})}

ANNUNCI DA VALUTARE:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

ISTRUZIONI:
- Usa il contesto mercato per dare un punteggio relativo, non assoluto.
- Un articolo sotto la mediana in buone condizioni merita 0.75+.
- Un articolo sopra la media senza vantaggi evidenti merita < 0.45.
- seller_trust 0–1 (più alto = più affidabile); None = non valutato (penalizzare leggermente).
- "reason" deve essere breve (max 10 parole, in italiano).

Rispondi SOLO con un array JSON valido:
[{{"id": "...", "value_score": 0.XX, "reason": "..."}}]"""

    try:
        response, _ = await call_llm(prompt)
        if not response:
            return {}
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                s["id"]: {"score": float(s["value_score"]), "reason": s.get("reason")}
                for s in data
                if "id" in s and "value_score" in s
            }
    except Exception as e:
        logger.warning("LLM value score computation failed: %s", e)
    return {}


def _merge_hybrid_results(ebay_items: List[Dict], memory_items: List[Dict], query: str = "", parsed: Dict = None) -> List[Dict]:
    """
    Merges live results from eBay with historical results from Global Memory (Qdrant).
    Uses a simplified RRF (Reciprocal Rank Fusion) logic.
    """
    parsed = parsed or {}
    product_word = (parsed.get("product") or "").lower()
    merged_map: Dict[str, Dict] = {}
    
    # 1. Process Memory results (historical)
    q_words = set(re.findall(r"\w+", query.lower())) - {"il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "a", "da", "con"}
    valid_q_words = [w for w in q_words if len(w) > 2]
    
    for i, item in enumerate(memory_items):
        eid = item.get("ebay_id")
        if not eid: continue
        
        title_low = (item.get("title") or "").lower()
        
        # Filtro forte 1: Se abbiamo il "product" target, verifichiamone la base (esclude plurali/suffissi lievi)
        if product_word and len(product_word) > 3:
            stem = product_word[:-1]
            if stem not in title_low:
                continue
                
        # Filtro forte 2: Almeno il 50% delle parole rilevanti della query deve matchare nel titolo
        if valid_q_words:
            match_count = sum(1 for w in valid_q_words if w in title_low)
            required = max(1, len(valid_q_words) // 2)
            if match_count < required:
                continue

        # Filtro forte 3: Prezzo (se presente nel parsed)
        price_val = item.get("price")
        if parsed and parsed.get("constraints") and price_val:
            try:
                p_float = float(price_val)
                price_ok = True
                for c in parsed["constraints"]:
                    if c.get("type") == "price":
                        op = c.get("operator")
                        val = c.get("value")
                        if op == "<=" and p_float > float(val):
                            price_ok = False
                            break
                        if op == ">=" and p_float < float(val):
                            price_ok = False
                            break
                        if op == "between" and isinstance(val, list):
                            if p_float < float(val[0]) or p_float > float(val[1]):
                                price_ok = False
                                break
                if not price_ok:
                    continue
            except Exception:
                pass

        # Filtro forte 4: Condizione (se presente nel parsed)
        if parsed and parsed.get("constraints"):
            item_cond = (item.get("condition") or "").lower()
            cond_ok = True
            for c in parsed["constraints"]:
                if c.get("type") == "condition":
                    target_cond = str(c.get("value", "")).lower()
                    if target_cond == "new" and not any(w in item_cond for w in ["nuovo", "new"]):
                        cond_ok = False
                        break
                    if target_cond == "used" and not any(w in item_cond for w in ["usato", "used", "gebraucht"]):
                        cond_ok = False
                        break
            if not cond_ok:
                continue

        item["_source_memory"] = True
        item["_memory_rank"] = i + 1
        merged_map[eid] = item
        
    # 2. Process eBay results (live)
    for i, item in enumerate(ebay_items):
        eid = item.get("ebay_id")
        if not eid: continue
        
        if eid in merged_map:
            # Item found in both! Give a strong relevance bonus
            existing = merged_map[eid]
            existing["_source_ebay"] = True
            existing["_ebay_rank"] = i + 1
            existing["_hybrid_bonus"] = 0.15 # Strong signal
            # Update with live price/rating if newer
            existing["price"] = item.get("price") or existing.get("price")
            existing["seller_rating"] = item.get("seller_rating") or existing.get("seller_rating")
        else:
            item["_source_ebay"] = True
            item["_ebay_rank"] = i + 1
            merged_map[eid] = item
            
    # Convert back to list
    final_list = list(merged_map.values())
    
    # Sort by a combination of original ranks and hybrid bonus
    def sort_key(x):
        # Lower is better for ranks, higher is better for bonus
        ebay_r = x.get("_ebay_rank", 100)
        mem_r = x.get("_memory_rank", 100)
        bonus = x.get("_hybrid_bonus", 0)
        # Final weight: prefer eBay top results, but boost those found in memory
        return (1.0 / (ebay_r + 10)) + (0.5 / (mem_r + 10)) + bonus

    final_list.sort(key=sort_key, reverse=True)
    return final_list


async def run_search_pipeline(
    query: str,
    db: Session,
    user: Optional[object] = None,
    llm_engine: str = "gemini",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Query vuota")

    logger.debug("PIPELINE START query='%s' | user=%s", query, getattr(user, 'id', 'anonymous'))
    logger.info("PIPELINE START for query: '%s'", query)
    t0 = time.time()
    timings = {}

    # ============================================================
    # 0) FETCH FROM GLOBAL MEMORY (PHANTOM STEP)
    # ============================================================
    logger.info("PIPELINE STEP 0: global_memory_fetch")
    t = time.time()
    memory_items_task = asyncio.create_task(asyncio.to_thread(retrieve_context, query, k=15, doc_type="product"))
    
    # ============================================================
    # 1) PARSE & EBAY SEARCH
    # ============================================================

    logger.info("PIPELINE STEP 1: parse_query")

    # Recupera il contesto se disponibile (session_id or user_id)
    context_info = ""
    target_id = session_id or (str(getattr(user, "id", "")) if user else None)
    if target_id:
        history = redis_client.get_user_queries(target_id)
        if history:
            # Prendi le ultime 3 per non appesantire troppo il prompt
            context_info = " | ".join(history[:3])
            logger.info("PIPELINE: Found context in history: %s", context_info)

    t = time.time()
    parsed = await parse_query_service(
        query,
        use_llm=(llm_engine != "rule_based"),
        include_meta=True,
        context_info=context_info,
    )
    t_parse = time.time() - t
    timings["parse_query_s"] = int(float(t_parse) * 1000) / 1000.0

    ebay_query_used = _build_ebay_query(parsed, query)

    # ============================================================
    # 2) PARALLEL: EBAY SEARCH + RAG RETRIEVAL (including expansion)
    # ============================================================
    logger.info("PIPELINE: Parallel Search & RAG")
    t = time.time()

    async def _do_rag():
        try:
            expanded = await expand_query(query)
            docs = await asyncio.to_thread(retrieve_context, expanded, k=10)
            return expanded, docs
        except Exception as e:
            logger.warning("RAG retrieve failed: %s", e)
            return query, []

    # Run heavy I/O in parallel
    results = await asyncio.gather(
        search_items(parsed, limit=MAX_RESULTS_FROM_EBAY),
        _do_rag()
    )
    search_res = results[0] if isinstance(results[0], dict) else {}
    items = search_res.get("itemSummaries", [])
    aspect_distributions = search_res.get("aspectDistributions", [])
    category_distributions = search_res.get("categoryDistributions", [])
    
    # -------------------------------------------------------------------------
    # SMART CATEGORY FILTERING (Anti-Accessory Logic)
    # -------------------------------------------------------------------------
    intent_type = parsed.get("product_type", "Unknown")
    
    if intent_type == "Main" and items:
        accessory_cats = ["Accessori", "Parti", "Cover", "Case", "Protective", "Replacement", "Cavo", "Caricatore"]
        
        # Check top 5 items for accessory categories
        top_items = [items[i] for i in range(len(items)) if i < 5]
        acc_count = 0
        for it in top_items:
            cat_name = it.get("category_name") or ""
            if any(word.lower() in cat_name.lower() for word in accessory_cats):
                acc_count += 1
        
        # If > 60% are accessories but we want Main
        if acc_count >= 3:
            logger.info("SMART FILTER | Detected accessory dominance (%d/5) for Main intent. Attempting re-search.", acc_count)
            
            best_cat_id = None
            for dist in category_distributions:
                c_name = dist.get("categoryName") or ""
                if not any(word.lower() in c_name.lower() for word in accessory_cats):
                    best_cat_id = dist.get("categoryId")
                    logger.info("SMART FILTER | Found better category: '%s' (%s)", c_name, best_cat_id)
                    break
            
            if best_cat_id:
                logger.info("SMART FILTER | Triggering RE-SEARCH with categoryId=%s", best_cat_id)
                # Modify a copy of parsed query to add the category filter
                parsed_copy = parsed.copy()
                constraints = list(parsed_copy.get("constraints") or [])
                constraints.append({"type": "category_id", "value": best_cat_id})
                parsed_copy["constraints"] = constraints
                
                new_search_res = await search_items(parsed_copy, limit=MAX_RESULTS_FROM_EBAY)
                items = new_search_res.get("itemSummaries", [])
                aspect_distributions = new_search_res.get("aspectDistributions", [])
                category_distributions = new_search_res.get("categoryDistributions", [])

    logger.info("Final eBay results count: %d", len(items))
    expanded_query, rag_docs = results[1]
    
    delta_io = time.time() - t
    timings["parallel_io_s"] = int(float(delta_io) * 1000) / 1000.0

    # 1.5) MERGE WITH GLOBAL MEMORY
    try:
        memory_items = await memory_items_task
        if memory_items:
            logger.info("Merging %d items from Global Memory", len(memory_items))
            items = _merge_hybrid_results(items, memory_items, query=query, parsed=parsed)
            timings["memory_hit"] = True
    except Exception as e:
        logger.warning("Memory fetch failed: %s", e)

    items = _dedupe_items(items) if items else []

    # ============================================================
    # 2.5 PROACTIVE METADATA ENRICHMENT
    # ============================================================
    dominant_category = None
    if items:
        # Trova la categoria più frequente tra i risultati
        cats = [it.get("categoryId") for it in items if it.get("categoryId")]
        if cats:
            dominant_category = max(set(cats), key=cats.count)
            logger.info("PIPELINE: Detected dominant category %s", dominant_category)

    # Inizia il fetch delle policy in background
    meta_task = None
    if dominant_category:
        meta_task = asyncio.create_task(get_return_policies(category_id=dominant_category))

    # Separate RAG docs for reranker
    product_docs = [d for d in rag_docs if d.get("type") == "product"]
    seller_docs = [d for d in rag_docs if d.get("type") == "seller_feedback"]

    # ============================================================
    # 3) USER PROFILE AUTO-UPDATE (coexists with manual MCP overrides)
    # ============================================================
    # --- Dominant Category Extraction (Common Context) ---
    _dominant_cat_name: Optional[str] = None
    if items:
        cat_names = [it.get("category_name") for it in items if it.get("category_name")]
        if cat_names:
            _dominant_cat_name = max(set(cat_names), key=cat_names.count)

    if user:
        try:
            changed = update_user_profile(
                user=user,
                parsed=parsed,
                db=db,
                dominant_category_name=_dominant_cat_name,
                items=items[:10] if items else None,
                action="search",
            )
            if changed:
                logger.info("Auto-profiling updated user %s (cat=%s)", getattr(user, 'id', '?'), _dominant_cat_name)
        except Exception as _prof_err:
            logger.warning("User auto-profiling update failed: %s", _prof_err)

    # ============================================================
    # 4) SELLER TRUST  (deve venire PRIMA del rerank per alimentarlo)
    # ============================================================
    logger.info("PIPELINE STEP 5: seller_trust")
    t = time.time()
    seller_trust_map = await _prefetch_top_sellers_feedback([items[i] for i in range(len(items)) if i < MAX_SELLERS_FOR_TRUST * 2])
    delta_trust_time = time.time() - t
    timings["seller_trust_s"] = int(float(delta_trust_time) * 1000) / 1000.0

    # Inietta trust_score negli item prima del reranker
    for item in items:
        seller_name = item.get("seller_name")
        item["trust_score"] = seller_trust_map.get(seller_name) if seller_name else None

    # ============================================================
    # 5) RERANK  (ora ha trust_score disponibile per ogni item)
    # ============================================================
    logger.info("PIPELINE STEP 6: rerank")

    t = time.time()
    if items:
        try:
            items = await asyncio.to_thread(
                rerank_products,
                query, items, user=user,
                product_docs=product_docs,
                seller_docs=seller_docs,
                constraints=parsed.get("constraints"),
            )
        except Exception as e:
            logger.exception("Rerank failed: %s", e)
            logger.warning("Rerank failed, keeping original order")
    delta_rerank = time.time() - t
    timings["rerank_s"] = int(float(delta_rerank) * 1000) / 1000.0

    # ============================================================
    # 5.5) NER ATTRIBUTE EXTRACTION
    # ============================================================
    logger.info("PIPELINE STEP 6.5: ner_extraction")
    t = time.time()
    if items:
        # Extract attributes for top few items only for performance
        top_n = 6
        top_titles = [items[i].get("title", "") for i in range(len(items)) if i < top_n]
        try:
            ner_results = await extract_attributes_batch(top_titles)
            top_items_ner = [items[i] for i in range(len(items)) if i < top_n]
            for it, ner in zip(top_items_ner, ner_results):
                it["ner_attributes"] = ner
        except Exception as e:
            logger.warning("NER extraction skipped: %s", e)
    delta_ner = time.time() - t
    timings["ner_extraction_s"] = int(float(delta_ner) * 1000) / 1000.0

    # ============================================================
    # 6) DB PERSIST (Isolated Session)
    # ============================================================
    logger.info("PIPELINE STEP 7: db_persist")
    t = time.time()
    
    # Use a localized session factory to avoid closure conflicts with the main request task
    def _bg_persist_thread_task():
        from app.db.database import SessionLocal
        bg_db = SessionLocal()
        try:
            # Prepare and persist items in an isolated session
            res, count = _prepare_and_persist_items(bg_db, items, seller_trust_map)
            bg_db.commit()
            return res, count
        except Exception as bg_e:
            logger.error("Background persistence error: %s", bg_e)
            bg_db.rollback()
            return [], 0
        finally:
            bg_db.close()

    results_out, saved_count = await asyncio.to_thread(_bg_persist_thread_task)
    delta_persist = time.time() - t
    timings["db_prepare_s"] = int(float(delta_persist) * 1000) / 1000.0

    # ============================================================
    # 7) FINAL RANKING + CONTEXT
    # ============================================================
    logger.info("PIPELINE STEP 8: final_ranking")

    _apply_final_ranking(results_out, user=user)

    # ------------------------------------------------------------
    # 7.5) LLM VALUATION (TOP-K REFINEMENT)
    # ------------------------------------------------------------
    logger.info("PIPELINE STEP 8.5: llm_value_refinement")
    top_n_for_llm = results_out[:8]
    if top_n_for_llm:
        llm_valuations = await _compute_llm_value_scores(
            top_items=top_n_for_llm,
            all_items=results_out,  # Mercato = intera SERP eBay corrente, zero costi extra
        )
        for item in results_out:
            eid = item.get("ebay_id")
            if eid and eid in llm_valuations:
                val = llm_valuations[eid]
                item["value_score"] = val["score"]
                if val.get("reason"):
                    explanations = list(item.get("explanations") or [])
                    item["explanations"] = [f"💡 {val['reason']}"] + explanations[:5]

    
    # Attach RAG feedback
    logger.info("PIPELINE STEP 9: rag_attach")
    for item in [results_out[i] for i in range(len(results_out)) if i < 15]:
        seller_name = item.get("seller_name")
        item["rag_feedback"] = [rag_docs[j] for j in range(len(rag_docs)) if j < 3 and rag_docs[j].get("seller") == seller_name] if seller_name else []

    # Recupera i risultati del fetch meta se completato
    meta_policies = None
    if meta_task:
        try:
            meta_policies = await meta_task
        except Exception:
            pass

    rag_context_text = build_context(query, results_out, rag_docs)
    
    if meta_policies and "returnPolicies" in meta_policies:
        # Aggiungi info sulle policy ai primi risultati
        policy = meta_policies["returnPolicies"][0] if meta_policies["returnPolicies"] else {}
        if policy:
          accepted = "accettato" if policy.get("returnsAccepted") else "non accettato"
          period = f"entro {policy.get('returnPeriod', {}).get('value')} {policy.get('returnPeriod', {}).get('unit')}" if policy.get("returnPeriod") else ""
          extra_info = f"\n[INFO EBAY] In questa categoria ({dominant_category}), il reso è generalmente {accepted} {period}."
          rag_context_text += extra_info
          logger.info("PIPELINE: Enriched RAG context with return policies")

    # ============================================================
    # 8) EXPLAIN (if results exist)
    # ============================================================
    logger.info("PIPELINE STEP 10: explain")

    t = time.time()
    analysis = None
    if results_out:
        try:
            analysis = await asyncio.to_thread(explain_results, query, [results_out[i] for i in range(len(results_out)) if i < 3])
        except Exception:
            pass
    delta_explain = time.time() - t
    timings["explain_s"] = int(float(delta_explain) * 1000) / 1000.0

    # ============================================================
    # 9) IR METRICS
    # ============================================================

    binary_relevance = [
        1 if (item.get("ranking_score", 0) >= 0.75) else 0
        for item in results_out
    ]

    metrics = {
        "precision@5": precision_at_k(binary_relevance, 5),
        "precision@10": precision_at_k(binary_relevance, 10),
        "recall@10": recall_at_k(
            binary_relevance,
            total_relevant=sum(binary_relevance),
            k=10
        ),
        "ndcg@10": ndcg_at_k(binary_relevance, 10),
    }

    # ============================================================
    # FINAL COMMIT
    # ============================================================

    try:
        # Commit can be slow, offload to thread
        await asyncio.to_thread(db.commit)
    except Exception:
        db.rollback()
        logger.warning("DB commit failed; transaction rolled back")

    # ============================================================
    # 11) PERSIST TO GLOBAL MEMORY (ASYNC)
    # ============================================================
    # Index in background to not block response
    asyncio.create_task(asyncio.to_thread(index_search_items, [results_out[i] for i in range(len(results_out)) if i < 15]))

    timings["total_s"] = int(float(time.time() - t0) * 1000) / 1000.0

    # Sanitize EVERYTHING before returning to ensure JSON serializability
    results_out = _sanitize_floats(results_out)
    metrics = _sanitize_floats(metrics)
    
    logger.debug("PIPELINE DONE | results=%d | time=%ss", len(results_out), timings['total_s'])

    return {
        "parsed_query": parsed,
        "ebay_query_used": ebay_query_used,
        "results_count": len(results_out),
        "saved_new_count": saved_count,
        "analysis": analysis,
        "results": results_out,
        "aspect_distributions": aspect_distributions,
        "dominant_category_name": _dominant_cat_name,
        "rag_context": rag_context_text,
        "metrics": metrics,
        "_timings": timings,
    }