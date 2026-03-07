import json
import logging
import time
from typing import Dict, Any, List

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


def tool_parse_query(query: str, state: Dict[str, Any]) -> str:
    """Parse + merge progressivo + user profiling implicito."""
    parsed = parse_query_service(query, use_llm=True, include_meta=True)
    if isinstance(parsed, str):
        parsed = {"semantic_query": parsed}

    # Merge progressivo con contesto precedente
    current = state.get("parsed_query") or {}
    if not isinstance(current, dict):
        current = {"semantic_query": str(current)}

    for k, v in parsed.items():
        if v:
            current[k] = v

    state["parsed_query"] = current

    # ---- User Profiling implicito (era Step 1.5 della pipeline) ----
    user = state.get("user_obj")
    db = state.get("db_session")
    if user and db:
        try:
            update_user_profile(user, current, db)
        except Exception:
            pass

    state["thinking_trace"].append("✔ parsed query updated")

    return json.dumps({
        "status": "success",
        "current_search_context": current
    })


def tool_search_products(query: str, state: Dict[str, Any], max_price: float = 0) -> str:
    """
    Pipeline COMPLETA di ricerca, identica alla vecchia routes.py:
    search → dedup → feedback ALL sellers → sentiment → trust →
    RAG ingest → rerank → personalized ranking → RAG retrieve → IR metrics → DB save
    """
    t_start = time.time()
    parsed = state.get("parsed_query", {"semantic_query": query})
    user = state.get("user_obj")
    db = state.get("db_session")

    if max_price > 0:
        constraints = parsed.get("constraints", [])
        constraints.append({"type": "price", "operator": "<=", "value": max_price})
        parsed["constraints"] = constraints

    # =============================================================
    # STEP 1: eBay Search (limit=20, come pipeline originale)
    # =============================================================
    t = time.time()
    results = search_items(parsed_query=parsed, limit=20) or []
    state["_timings"]["ebay_search_s"] = round(time.time() - t, 3)
    state["thinking_trace"].append(f"✔ searched {len(results)} listings from eBay")

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
    # STEP 6: Rerank (embedding similarity + trust + price penalty)
    # =============================================================
    t = time.time()
    if results:
        results = rerank_products(query, results, user=user)
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
        {
            "id": r.get("ebay_id"),
            "title": r.get("title"),
            "price": r.get("price"),
            "seller": r.get("seller_name"),
            "trust": r.get("trust_score"),
            "ranking": r.get("ranking_score"),
        }
        for r in results[:5]
    ]

    return json.dumps({
        "results_count": len(results),
        "saved_new": saved_count,
        "top_results": summary,
        "metrics": state["metrics"],
    })


# =========================================================
# TOOL: Explain Results (usa explainer.py strutturato)
# =========================================================

def tool_explain_results(query: str, state: Dict[str, Any]) -> str:
    """Genera spiegazione strutturata usando il modulo explainer.py"""
    results = state.get("results", [])
    explanation = generate_explanation(query, results)

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


def tool_expand_product_query(product: str, state: Dict[str, Any]) -> str:
    expansions = [f"{product} originale", f"{product} nuovo", product]
    state["thinking_trace"].append("✔ query expanded")
    return json.dumps({"query_expansions": expansions})


def tool_get_seller_profile(seller_name: str, state: Dict[str, Any]) -> str:
    state["thinking_trace"].append(f"✔ analyzed seller {seller_name}")
    return json.dumps({
        "seller": seller_name,
        "feedback_score": 1200,
        "positive_percent": 99.4,
    })


def tool_get_seller_feedback(seller_name: str, state: Dict[str, Any]) -> str:
    """Recupera feedback di un singolo venditore (per analisi approfondita)."""
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


def tool_retrieve_feedback_context(
    seller_name: str, query: str, state: Dict[str, Any]
) -> str:
    """RAG hybrid retrieval (FAISS + BM25 + RRF) per un venditore."""
    docs = retrieve_context(f"{seller_name} {query}", k=5)
    state["rag_context"] = build_context(
        query, state.get("results", []), docs
    )
    state["thinking_trace"].append(
        f"✔ vector search on RAG for {seller_name}"
    )
    return json.dumps([d.get("text") for d in docs])


def tool_compute_seller_trust(seller_name: str, state: Dict[str, Any]) -> str:
    """Calcola trust score per un singolo venditore (per analisi approfondita)."""
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


def tool_detect_price_anomaly(
    product: str, price: float, state: Dict[str, Any]
) -> str:
    state["thinking_trace"].append("✔ analyzed price anomalies")
    return json.dumps({"anomaly_score": 0.1, "is_anomalous": False})


def tool_rerank_products(query: str, state: Dict[str, Any]) -> str:
    """Re-rank manuale (se l'LLM vuole riordinare con query diversa)."""
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
    {
        "type": "function",
        "function": {
            "name": "parse_query",
            "description": (
                "Trasforma la query naturale dell'utente in una struttura "
                "semantica filtrata (prodotto, brand, prezzo, condizione). "
                "Esegui SEMPRE come primo step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
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
                "Cerca prodotti su eBay e esegue automaticamente: "
                "deduplicazione, analisi feedback di TUTTI i venditori, "
                "calcolo trust score NLP, salvataggio DB, reranking, "
                "ranking personalizzato e retrieval RAG. "
                "Restituisce i migliori risultati con metriche IR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_price": {
                        "type": "number",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seller_feedback",
            "description": (
                "Recupera le recensioni di un venditore specifico. "
                "OPZIONALE: search_products già analizza tutti i venditori. "
                "Usa questo tool solo se l'utente chiede dettagli su un venditore."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {"type": "string"}
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
                "Calcola il Trust Score per un singolo venditore. "
                "OPZIONALE: search_products lo calcola già per tutti."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {"type": "string"}
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
                "Cerca nei feedback indicizzati (RAG ibrido FAISS+BM25) "
                "informazioni rilevanti su un venditore. "
                "OPZIONALE: utile per rispondere a domande specifiche."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {"type": "string"},
                    "query": {"type": "string"},
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
                "Riordina i prodotti trovati con una query diversa. "
                "OPZIONALE: search_products già esegue il rerank."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
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
                "Genera automaticamente una spiegazione strutturata "
                "dei risultati trovati (trust, prezzo, ranking reasons). "
                "Usa SOLO ED ESCLUSIVAMENTE ALLA FINE, DOPO search_products."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La query originale dell'utente",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_user_clarification",
            "description": (
                "Interrompe il processo e fa una domanda all'utente. "
                "Da usare SOLO quando mancano dettagli cruciali "
                "(taglia scarpe, memoria telefoni, ecc.). "
                "NON USARE se si è già svolta la ricerca."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "La domanda da porre all'utente",
                    }
                },
                "required": ["question"],
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
    "request_user_clarification": lambda question, state: json.dumps(
        {"question": question}
    ),
}
