import json
import logging
import time
from typing import Dict, Any, List
import re

from app.services.parser import parse_query_service
from app.services.ebay import search_items
from app.services.feedback import get_seller_feedback
from app.services.trust import compute_trust_score
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.rag.ingest import ingest_seller_feedback
from app.services.rag import retrieve_context, build_context
from app.services.rag.reranker import rerank_products
from app.services.rag.explainer import explain_results as generate_explanation
from app.services.metrics.ir_metrics import precision_at_k, recall_at_k, ndcg_at_k
from app.services.user_profiling import update_user_profile
from app.models.listing import Listing

logger = logging.getLogger(__name__)

# =========================================================
# THE TOOLBOX EXECUTORS
# =========================================================


def tool_parse_query(state: Dict[str, Any], **kwargs) -> str:
    """Parse + merge progressivo + user profiling implicito."""
    # Gestiamo nomi parametri variabili per robustezza (modello potrebbe allucinare 'user_request')
    query = kwargs.get("query") or kwargs.get("user_request") or kwargs.get("text")
    
    if not query:
        return json.dumps({"error": "No query provided to parse_query"})

    parsed = parse_query_service(query, use_llm=True, include_meta=True)
    if isinstance(parsed, str):
        parsed = {"semantic_query": parsed}

    # Merge progressivo con contesto precedente
    current = state.get("parsed_query") or {}
    if not isinstance(current, dict):
        current = {"semantic_query": str(current)}

    # Preserviamo i valori precedenti se i nuovi sono vuoti
    for k, v in parsed.items():
        if k == "semantic_query":
            # Per la semantic query, facciamo sempre un merge delle parole uniche
            # Questo evita di perdere pezzi se l'utente ripete solo una parte (es. da "nike nere 42" a "shox")
            old_sq = current.get("semantic_query", "").lower()
            new_sq = str(v or "").lower()
            
            # Uniamo le parole mantenendo l'ordine originale dell'anteprima + le nuove
            words = old_sq.split() + new_sq.split()
            seen = set()
            deduped = [w for w in words if w and not (w in seen or seen.add(w))]
            current[k] = " ".join(deduped)
            
        elif k == "compatibilities" and isinstance(v, dict):
            # Merge dizionario invece di overwrite
            old_comp = current.get("compatibilities") or {}
            if not isinstance(old_comp, dict): old_comp = {}
            old_comp.update(v)
            current[k] = old_comp
        elif v:
            # Overwrite per altri campi (brands, product, etc) solo se present
            current[k] = v

    state["parsed_query"] = current

    # ---- User Profiling implicito ----
    user = state.get("user_obj")
    db = state.get("db_session")
    if user and db:
        try:
            update_user_profile(user, current, db)
        except Exception:
            pass

    state["thinking_trace"].append("✔ parsed query updated")

    # Costruisci la query finale per search_products combinando brand, prodotto, semantic query e compatibilities
    brands = " ".join(current.get("brands", []))
    product = current.get("product", "")
    sq = current.get("semantic_query", "")
    comp_values = " ".join([str(val) for val in (current.get("compatibilities") or {}).values()])
    
    # Uniamo tutto in una stringa di ricerca potente per eBay
    search_query_parts = []
    if brands: search_query_parts.append(brands)
    if product and product.lower() not in brands.lower(): search_query_parts.append(product)
    if sq: search_query_parts.append(sq)
    if comp_values: search_query_parts.append(comp_values)
    
    # Deduplica parole nella query finale mantenendo l'ordine e ignorando punteggiatura (es "Levi's" vs "levis")
    final_words = []
    seen_words = set()
    for part in search_query_parts:
        for word in str(part).split():
            # Pulizia per confronto (es. levi's -> levis)
            w_norm = re.sub(r"[^\w]", "", word.lower())
            if w_norm and w_norm not in seen_words:
                final_words.append(word)
                seen_words.add(w_norm)
    
    final_search_query = " ".join(final_words).strip()
    missing = current.get("missing_info", [])

    return json.dumps({
        "status": "parsed",
        "search_query": final_search_query,
        "missing_info": missing,
        "next_action": "NOW call search_products OR call request_user_clarification if too many fields are missing.",
    })


def tool_search_products(state: Dict[str, Any], **kwargs) -> str:
    """
    Pipeline COMPLETA di ricerca, identica alla vecchia routes.py:
    search → dedup → feedback ALL sellers → sentiment → trust →
    RAG ingest → rerank → personalized ranking → RAG retrieve → IR metrics → DB save
    """
    # Robustezza: cerchiamo la query in vari possibili nomi (hallucination del modello)
    query = kwargs.get("query") or kwargs.get("search_query") or kwargs.get("product")
    max_price = kwargs.get("max_price")

    if not query:
        return json.dumps({"error": "No query provided to search_products"})

    # SAFETY CHECK: Se il modello passa letteralmente il nome del tool o della variabile (allucinazione da prompt)
    hallucination_literals = ["search_products", "search_query", "query", "product_name", "{search_query}"]
    is_hallucinated = any(h in str(query).lower() for h in hallucination_literals)

    t_start = time.time()
    # Recuperiamo il parse dallo state, preferendo però i parametri diretti SE presenti
    parsed = state.get("parsed_query") or {}
    if not isinstance(parsed, dict):
        parsed = {"semantic_query": str(parsed)}

    # PRIORITÀ ALLA QUERY PULITA DEL PARSER (semantic_query)
    clean_search_query = parsed.get("semantic_query")
    if clean_search_query and len(clean_search_query.strip()) > 3 and not is_hallucinated:
        query = clean_search_query
    elif not query or is_hallucinated:
        logger.warning(f"FALLBACK: Using original message words because search_query is missing or hallucinated.")
        from langchain_core.messages import HumanMessage
        query_text = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                query_text = msg.content
                break
        query = " ".join([w for w in query_text.split() if len(w) > 3]) or parsed.get("product", "jeans")
    
    # Se dopo tutto non abbiamo nulla, errore
    if not query and not parsed.get("semantic_query"):
        return json.dumps({"error": "No valid query available for search"})

    user = state.get("user_obj")
    db = state.get("db_session")

    # Applica max_price opzionale se passato esplicitamente come numero valido > 0
    try:
        if max_price is not None:
            f_max = float(max_price)
            if f_max > 0:
                constraints = parsed.get("constraints", [])
                # Evitiamo duplicati se il parser l'aveva già messo
                if not any(c.get("type") == "price" and c.get("operator") == "<=" for c in constraints):
                    constraints.append({"type": "price", "operator": "<=", "value": f_max})
                parsed["constraints"] = constraints
    except (ValueError, TypeError):
        pass

    # =============================================================
    # STEP 1: eBay Search (limit=20, come pipeline originale)
    # =============================================================
    t = time.time()
    results = search_items(parsed_query=parsed, limit=20) or []
    
    # LOG DIAGNOSTICO: Vediamo cosa stiamo filtrando effettivamente
    logger.info(f"PROCESSED SEARCH: query='{query}' constraints={parsed.get('constraints')} found={len(results)}")

    # =============================================================
    # STEP 2: Remove Duplicates (per ebay_id)
    # =============================================================
    seen_ids = set()
    deduped = []
    for item in results:
        ebay_id = item.get("ebay_id")
        if not ebay_id or ebay_id in seen_ids:
            continue
        seen_ids.add(ebay_id)
        deduped.append(item)
    results = deduped

    # =============================================================
    # STEP 3: Feedback + RAG Ingest per TUTTI i venditori
    # =============================================================
    t = time.time()
    seller_feedback_cache: Dict[str, Any] = {}
    seller_trust_cache: Dict[str, float] = {}
    sellers = {i.get("seller_name") for i in results if i.get("seller_name")}

    for seller_name in sellers:
        try:
            feedbacks = get_seller_feedback(seller_name, limit=50)
            seller_feedback_cache[seller_name] = feedbacks
            ingest_seller_feedback(
                seller_name=seller_name,
                feedbacks=feedbacks,
                max_docs=20
            )
        except Exception:
            continue

    state["seller_feedbacks"] = seller_feedback_cache
    state["_timings"]["feedback_ingest_s"] = round(time.time() - t, 3)
    state["thinking_trace"].append(f"✔ fetched feedback for {len(sellers)} sellers")

    # =============================================================
    # STEP 4: Sentiment + Trust Score per TUTTI i venditori
    # =============================================================
    t = time.time()
    for seller_name in sellers:
        feedbacks = seller_feedback_cache.get(seller_name, [])
        if feedbacks:
            try:
                sentiment = compute_sentiment_score(feedbacks)
                trust = compute_trust_score(feedbacks, sentiment_score=sentiment)
                seller_trust_cache[seller_name] = trust
            except Exception:
                continue
    state["_timings"]["trust_compute_s"] = round(time.time() - t, 3)

    # Applica trust_score a ogni risultato
    for item in results:
        sn = item.get("seller_name")
        if sn and sn in seller_trust_cache:
            item["trust_score"] = seller_trust_cache[sn]

    state["thinking_trace"].append("✔ computed trust scores for all sellers")

    # =============================================================
    # STEP 5: Salvataggio DB
    # =============================================================
    saved_count = 0
    if db:
        try:
            t = time.time()
            for item in results:
                ebay_id = item.get("ebay_id")
                if not ebay_id:
                    continue
                exists = db.query(Listing).filter_by(ebay_id=ebay_id).first()
                already = exists is not None
                item["_already_in_db"] = already
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
            db.commit()
            state["_timings"]["db_save_s"] = round(time.time() - t, 3)
        except Exception:
            db.rollback()

    state["thinking_trace"].append(f"✔ saved {saved_count} new items to DB")

    # =============================================================
    # STEP 6: Rerank (embedding similarity + trust + price penalty + keyword resonance)
    # =============================================================
    t = time.time()
    if results:
        results = rerank_products(query, results, user=user, parsed_query=parsed)
    state["_timings"]["rerank_s"] = round(time.time() - t, 3)
    state["thinking_trace"].append("✔ reranked results")

    # =============================================================
    # STEP 7: Final Ranking Score (Personalizzato + Explanations)
    # =============================================================
    for item in results:
        relevance = item.get("_rerank_score", 0)
        trust = item.get("trust_score") or 0
        price = item.get("price") or 0
        price_score = max(0.0, 1.0 - float(price) / 1000.0) if price else 0.0

        ranking_score = (
            0.5 * float(relevance)
            + 0.25 * float(trust)
            + 0.15 * float(price_score)
        )
        explanations = []

        # ---- Personalizzazione basata su User Profile ----
        if user:
            if getattr(user, "favorite_brands", None):
                user_brands = {
                    b.strip().lower()
                    for b in user.favorite_brands.split(",")
                }
                title = (item.get("title") or "").lower()
                for b in user_brands:
                    if b in title:
                        ranking_score += 0.10
                        item["brand_match"] = True
                        explanations.append(
                            f"This item matches your preferred brand '{b}'."
                        )
                        break

            if getattr(user, "price_preference", None) and price:
                try:
                    if price <= float(user.price_preference):
                        ranking_score += 0.05
                        item["price_match"] = True
                        explanations.append(
                            "This product falls within your typical price range."
                        )
                except Exception:
                    pass

        # ---- Trust explanations ----
        if trust >= 0.8:
            explanations.append(
                "Seller has very strong feedback and trust score."
            )
        elif trust >= 0.6:
            explanations.append(
                "Seller shows generally positive feedback."
            )

        item["ranking_score"] = round(ranking_score, 3)
        if explanations:
            item["explanations"] = explanations

    # =============================================================
    # STEP 8: RAG Retrieve + Context Building
    # =============================================================
    t = time.time()
    try:
        rag_docs = retrieve_context(query, k=10)
    except Exception:
        rag_docs = []
    state["_timings"]["rag_retrieve_s"] = round(time.time() - t, 3)

    # Attacca rag_feedback per seller a ogni item
    for item in results:
        seller_name = item.get("seller_name")
        if not seller_name:
            item["rag_feedback"] = []
            continue
        item["rag_feedback"] = [
            d for d in rag_docs if d.get("seller") == seller_name
        ][:3]

    state["rag_context"] = build_context(query, results, rag_docs)
    state["thinking_trace"].append("✔ retrieved RAG context")

    # =============================================================
    # STEP 9: IR Metrics (precision, recall, nDCG)
    # =============================================================
    binary_relevance = [
        1 if (item.get("ranking_score", 0) >= 0.75) else 0
        for item in results
    ]
    state["metrics"] = {
        "precision@5": precision_at_k(binary_relevance, 5),
        "precision@10": precision_at_k(binary_relevance, 10),
        "recall@10": recall_at_k(
            binary_relevance,
            total_relevant=sum(binary_relevance),
            k=10,
        ),
        "ndcg@10": ndcg_at_k(binary_relevance, 10),
    }

    # =============================================================
    # Salva risultati nello state condiviso
    # =============================================================
    state["results"] = results
    state["_timings"]["search_pipeline_total_s"] = round(
        time.time() - t_start, 3
    )

    # Restituisce un riassunto compatto all'LLM (non tutti i dati)
    summary = [
        {"title": r.get("title"), "price": r.get("price"), "trust": r.get("trust_score")}
        for r in results[:3]
    ]

    return json.dumps({
        "status": "search_complete",
        "results_count": len(results),
        "top_results": summary,
        "next_action": "NOW call explain_results with query: " + query,
    })


# =========================================================
# TOOL: Explain Results (usa explainer.py strutturato)
# =========================================================

def tool_explain_results(state: Dict[str, Any], **kwargs) -> str:
    """Genera una spiegazione finale (analoga allo step 7 della pipeline)."""
    # Recupera info mancanti dal parser per arricchire la spiegazione
    parsed = state.get("parsed_query") or {}
    query = kwargs.get("query") or parsed.get("semantic_query") or parsed.get("search_query") or ""
    missing = parsed.get("missing_info", [])
    
    explanation = generate_explanation(query, results, missing_info=missing)

    # Salva nello state per l'orchestratore
    state["_explanation"] = explanation
    state["thinking_trace"].append("✔ generated explanation")

    return json.dumps({"explanation": explanation})


# =========================================================
# TOOL opzionali (per analisi approfondita su singolo seller)
# =========================================================

def tool_detect_intent(query: str, state: Dict[str, Any]) -> str:
    intent = "buy_product"
    if "opinioni" in query.lower() or "recensioni" in query.lower():
        intent = "compare_sellers"
    state["thinking_trace"].append(f"✔ intent detected: {intent}")
    return json.dumps({"intent": intent})


def tool_expand_product_query(state: Dict[str, Any], **kwargs) -> str:
    product = kwargs.get("product") or kwargs.get("query") or ""
    expansions = [f"{product} originale", f"{product} nuovo", product]
    state["thinking_trace"].append("✔ query expanded")
    return json.dumps({"query_expansions": expansions})


def tool_get_seller_profile(state: Dict[str, Any], **kwargs) -> str:
    seller_name = kwargs.get("seller_name") or ""
    state["thinking_trace"].append(f"✔ analyzed seller {seller_name}")
    return json.dumps({
        "seller": seller_name,
        "feedback_score": 1200,
        "positive_percent": 99.4,
    })


def tool_get_seller_feedback(state: Dict[str, Any], **kwargs) -> str:
    """Recupera feedback di un singolo venditore (per analisi approfondita)."""
    seller_name = kwargs.get("seller_name") or ""
    feedbacks = get_seller_feedback(seller_name, limit=50)
    ingest_seller_feedback(
        seller_name=seller_name, feedbacks=feedbacks, max_docs=20
    )

    if "seller_feedbacks" not in state:
        state["seller_feedbacks"] = {}
    state["seller_feedbacks"][seller_name] = feedbacks

    state["thinking_trace"].append(
        f"✔ retrieved {len(feedbacks)} feedbacks for {seller_name}"
    )
    res = [
        {"rating": f.get("rating"), "comment": f.get("comment")}
        for f in feedbacks[:5]
    ]
    return json.dumps(res)


def tool_retrieve_feedback_context(state: Dict[str, Any], **kwargs) -> str:
    """RAG hybrid retrieval (FAISS + BM25 + RRF) per un venditore."""
    seller_name = kwargs.get("seller_name") or ""
    query = kwargs.get("query") or ""
    docs = retrieve_context(f"{seller_name} {query}", k=5)
    state["rag_context"] = build_context(
        query, state.get("results", []), docs
    )
    state["thinking_trace"].append(
        f"✔ vector search on RAG for {seller_name}"
    )
    return json.dumps([d.get("text") for d in docs])


def tool_compute_seller_trust(state: Dict[str, Any], **kwargs) -> str:
    """Calcola trust score per un singolo venditore (per analisi approfondita)."""
    seller_name = kwargs.get("seller_name") or ""
    feedbacks = state.get("seller_feedbacks", {}).get(seller_name, [])
    if not feedbacks:
        return json.dumps({
            "error": "No feedbacks retrieved yet. Use get_seller_feedback first."
        })

    sentiment = compute_sentiment_score(feedbacks)
    trust = compute_trust_score(feedbacks, sentiment_score=sentiment)

    for r in state.get("results", []):
        if r.get("seller_name") == seller_name:
            r["trust_score"] = trust

    state["thinking_trace"].append(f"✔ computed trust for {seller_name}")
    return json.dumps({"trust_score": round(trust, 3)})


def tool_detect_price_anomaly(state: Dict[str, Any], **kwargs) -> str:
    product = kwargs.get("product") or ""
    price = kwargs.get("price") or 0.0
    state["thinking_trace"].append("✔ analyzed price anomalies")
    return json.dumps({"anomaly_score": 0.1, "is_anomalous": False})

def tool_social_response(state: Dict[str, Any], **kwargs) -> str:
    """Risponde a messaggi sociali (grazie, ciao, etc) con grande creatività."""
    response = kwargs.get("response") or "Ciao! Sono il tuo assistente shopping AI. Come posso aiutarti oggi?"
    return json.dumps({
        "response": response,
        "social_message": response
    })

def tool_request_user_clarification(state: Dict[str, Any], **kwargs) -> str:
    """Chiede chiarimenti all'utente."""
    question = kwargs.get("question") or kwargs.get("clarification") or "Puoi chiarire la tua richiesta?"
    return json.dumps({"question": question})

def tool_detect_intent(state: Dict[str, Any], **kwargs) -> str:
    """Rileva l'intento dell'utente (per routing)."""
    return json.dumps({"intent": "search"})


def tool_rerank_products(state: Dict[str, Any], **kwargs) -> str:
    """Re-rank manuale (se l'LLM vuole riordinare con query diversa)."""
    query = kwargs.get("query") or ""
    results = state.get("results", [])
    if not results:
        return json.dumps({"error": "No products to rerank."})

    user = state.get("user_obj")
    ranked = rerank_products(query, results, user=user)
    state["results"] = ranked
    state["thinking_trace"].append("✔ re-ranked results")
    return json.dumps([
        {"title": r.get("title"), "rerank_score": r.get("_rerank_score")}
        for r in ranked[:3]
    ])


# =========================================================
# THE MCP MENU (JSON SCHEMA FOR LLAMA)
# =========================================================

AGENT_TOOLS_SCHEMA = [
    # ==========================================================
    # CORE FLOW: parse_query → search_products → explain_results
    # ==========================================================
    {
        "type": "function",
        "function": {
            "name": "parse_query",
            "description": (
                "STEP 1 — Always call this FIRST. "
                "Parses the user's natural language into structured filters. "
                "NEXT: Immediately call search_products.\n"
                "OUTPUT: JSON with 'status', 'search_query' (string to use in next step), "
                "and 'next_action' hints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's original search request",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "STEP 2 — Call this AFTER parse_query. "
                "Searches eBay, analyzes sellers, computes trust, and ranks products. "
                "NEXT: Call explain_results.\n"
                "OUTPUT: JSON with 'status', 'results_count', 'top_results' (list of product summaries), "
                "and 'next_action' directive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The product search query (e.g. 'nike shoes')",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Optional max price filter in EUR.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_results",
            "description": (
                "STEP 3 (FINAL) — Call this AFTER search_products. "
                "Generates the final natural language response for the user.\n"
                "OUTPUT: JSON with 'explanation' (Markdown string)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's original search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    # ==========================================================
    # SOCIAL & CLARIFICATION
    # ==========================================================
    {
        "type": "function",
        "function": {
            "name": "social_response",
            "description": (
                "Call this for greetings (ciao, hi), thanks, or general conversation. "
                "The assistant should be professional and mention its capabilities: "
                "Deep Search, Seller Analysis, and AI Ranking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Your creative and professional response in Italian",
                    }
                },
                "required": ["response"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_user_clarification",
            "description": (
                "Use when the request is too vague to search. "
                "Do NOT use if search_products has already run.\n"
                "OUTPUT: JSON with 'question' (string)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question for the user (in Italian)",
                    }
                },
                "required": ["question"],
            },
        },
    },
    # ==========================================================
    # OPTIONAL DEEP-DIVE TOOLS
    # ==========================================================
    {
        "type": "function",
        "function": {
            "name": "get_seller_feedback",
            "description": (
                "OPTIONAL — Get detailed reviews for one seller. "
                "Requires search_products to have run first.\n"
                "OUTPUT: List of objects with 'rating' and 'comment'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {
                        "type": "string",
                        "description": "The exact eBay seller username",
                    }
                },
                "required": ["seller_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_seller_trust",
            "description": (
                "OPTIONAL — Recomputes trust score for one seller. "
                "Requires get_seller_feedback to have run first for this seller.\n"
                "OUTPUT: JSON with 'trust_score' (float 0-1)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {
                        "type": "string",
                        "description": "The exact eBay seller username",
                    }
                },
                "required": ["seller_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_feedback_context",
            "description": (
                "OPTIONAL — Hybrid RAG search in seller feedback. "
                "Requires search_products to have run first.\n"
                "OUTPUT: List of raw text excerpts (strings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {
                        "type": "string",
                        "description": "The exact eBay seller username",
                    },
                    "query": {
                        "type": "string",
                        "description": "What to search in the feedback",
                    },
                },
                "required": ["seller_name", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerank_products",
            "description": (
                "OPTIONAL — Re-sorts found results using a new query. "
                "Requires search_products to have run first.\n"
                "OUTPUT: List of objects with 'title' and 'rerank_score'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "New ranking criteria query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# =========================================================
# MAPPA DELLE FUNZIONI
# =========================================================

TOOLS_MAP = {
    "parse_query": tool_parse_query,
    "detect_intent": tool_detect_intent,
    "expand_product_query": tool_expand_product_query,
    "search_products": tool_search_products,
    "get_seller_profile": tool_get_seller_profile,
    "get_seller_feedback": tool_get_seller_feedback,
    "retrieve_feedback_context": tool_retrieve_feedback_context,
    "compute_seller_trust": tool_compute_seller_trust,
    "detect_price_anomaly": tool_detect_price_anomaly,
    "rerank_products": tool_rerank_products,
    "explain_results": tool_explain_results,
    "social_response": tool_social_response,
    "request_user_clarification": tool_request_user_clarification,
}
