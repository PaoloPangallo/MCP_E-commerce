import os
import logging
import asyncio
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

SERP_API_KEY = os.getenv("SERP_API_KEY")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def filter_outliers_iqr(prices: List[float], items: List[dict]) -> Tuple[List[float], List[dict]]:
    if len(prices) < 4:
        return prices, items

    q1 = statistics.quantiles(prices, n=4)[0]
    q3 = statistics.quantiles(prices, n=4)[2]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered_prices, filtered_items = [], []
    for p, it in zip(prices, items):
        if lower <= p <= upper:
            filtered_prices.append(p)
            filtered_items.append(it)

    return filtered_prices, filtered_items


def classify_trend(current: int, previous: int) -> str:
    if previous == 0:
        return "stabile"
    change = (current - previous) / previous
    if change > 0.30:
        return "forte crescita"
    if change > 0.10:
        return "in crescita"
    if change < -0.30:
        return "forte calo"
    if change < -0.10:
        return "in calo"
    return "stabile"


def extract_seller_breakdown(shopping_results: List) -> List[Dict[str, Any]]:
    sellers = []
    for item in shopping_results[:10]:
        seller = item.get("source") or item.get("seller") or "Sconosciuto"
        price = item.get("extracted_price")
        link = item.get("link") or item.get("product_link", "")
        title = item.get("title", "")
        thumbnail = item.get("thumbnail", "")
        if price and seller:
            sellers.append({
                "seller": seller,
                "price": price,
                "title": title[:60],
                "link": link,
                "thumbnail": thumbnail,
            })
    return sorted(sellers, key=lambda x: x["price"])


# ─────────────────────────────────────────────
# FETCH FUNCTIONS (Google Shopping & Trends)
# ─────────────────────────────────────────────

async def fetch_google_shopping(query: str, api_key: str) -> Dict[str, Any]:
    from serpapi import GoogleSearch

    params = {
        "engine": "google_shopping",
        "q": query,
        "hl": "it",
        "gl": "it",
        "api_key": api_key,
    }

    try:
        def _search():
            search = GoogleSearch(params)
            return search.get_dict()

        results = await asyncio.to_thread(_search)
        shopping_results = results.get("shopping_results", [])

        if not shopping_results:
            return {"status": "no_data", "message": "Nessun risultato shopping trovato."}

        prices: List[float] = []
        valid_items: List[Dict[str, Any]] = []

        for item in shopping_results:
            price = item.get("extracted_price")
            if price:
                prices.append(float(price))
                valid_items.append(item)
            else:
                price_str = item.get("price")
                if price_str:
                    clean_price = price_str.replace("€", "").replace("$", "").strip()
                    if "," in clean_price and "." in clean_price:
                        clean_price = clean_price.replace(".", "").replace(",", ".")
                    elif "," in clean_price:
                        clean_price = clean_price.replace(",", ".")
                    try:
                        p_val = float(clean_price)
                        prices.append(p_val)
                        item["extracted_price"] = p_val
                        valid_items.append(item)
                    except ValueError:
                        continue

        if not prices:
            return {"status": "no_data", "message": "Impossibile estrarre i prezzi."}

        prices, valid_items = filter_outliers_iqr(prices, valid_items)

        if not prices:
            return {"status": "no_data", "message": "Tutti i prezzi sono stati esclusi come outlier."}

        avg_price = sum(prices) / len(prices)
        std_dev = statistics.stdev(prices) if len(prices) > 1 else 0.0
        median_price = statistics.median(prices)

        if std_dev < avg_price * 0.10:
            price_consistency = "prezzi molto uniformi sul mercato"
        elif std_dev < avg_price * 0.25:
            price_consistency = "prezzi mediamente variabili"
        else:
            price_consistency = "prezzi molto variabili: confronta bene prima di acquistare"

        return {
            "status": "ok",
            "min_price": min(prices),
            "max_price": max(prices),
            "average_price": round(avg_price, 2),
            "median_price": round(median_price, 2),
            "std_dev": round(std_dev, 2),
            "price_range": round(max(prices) - min(prices), 2),
            "price_consistency": price_consistency,
            "samples": len(prices),
            "top_results": valid_items[:3],
            "seller_breakdown": extract_seller_breakdown(valid_items),
        }

    except Exception as e:
        logger.error(f"Error fetching Google Shopping data: {e}")
        return {"status": "error", "error": str(e)}


async def fetch_google_trends(query: str, api_key: str) -> Dict[str, Any]:
    from serpapi import GoogleSearch

    trend_query = query
    trend_query = re.sub(r'\d+(?:gb|tb)', '', trend_query, flags=re.IGNORECASE)
    trend_query = re.sub(r'\b[A-Z0-9]{5,}-\d+\b', '', trend_query, flags=re.IGNORECASE)
    trend_query = re.sub(r'\b(taglia|size)\s*[\da-z/]+\b', '', trend_query, flags=re.IGNORECASE)
    trend_query = re.sub(r'\(.*?\)', '', trend_query)
    trend_query = re.sub(
        r'\b(nuovo|usato|ricondizionato|originale|sbloccato|pari al nuovo|scarpe|scarpa)\b',
        '', trend_query, flags=re.IGNORECASE
    )
    trend_query = re.sub(r'[-\s]+', ' ', trend_query).strip()

    if len(trend_query.split()) < 2 and len(query.split()) > 3:
        trend_query = re.sub(r'\b[A-Z0-9]{5,}-\d+\b', '', query, flags=re.IGNORECASE).strip()

    params = {
        "engine": "google_trends",
        "q": trend_query,
        "data_type": "TIMESERIES",
        "api_key": api_key,
    }

    try:
        def _search():
            search = GoogleSearch(params)
            return search.get_dict()

        results = await asyncio.to_thread(_search)
        timeline = results.get("interest_over_time", {}).get("timeline_data", [])

        if not timeline:
            return {"status": "no_data", "message": "Nessun dato di trend trovato."}

        interest_graph = []
        for item in timeline:
            date_str = item.get("date", "")
            match = re.search(r'([A-Z][a-z]{2})\s+\d+,\s+(\d{4})', date_str)
            display_date = f"{match.group(1)} {match.group(2)[2:]}" if match else date_str[:6]

            for v in item.get("values", []):
                try:
                    val = int(v.get("extracted_value", 0))
                    interest_graph.append({"date": display_date, "value": val})
                except ValueError:
                    pass

        trend_values = [p["value"] for p in interest_graph]
        if not trend_values:
            return {"status": "no_data", "message": "Dati di trend vuoti."}

        current = trend_values[-1]
        previous = trend_values[0] if len(trend_values) > 1 else current
        trend_direction = classify_trend(current, previous)

        related_queries = [
            q.get("query")
            for q in results.get("related_queries", {}).get("top", [])
            if q.get("query")
        ]

        return {
            "status": "ok",
            "current_interest": current,
            "trend_direction": trend_direction,
            "data_points": len(trend_values),
            "interest_graph": interest_graph,
            "related_queries": related_queries[:5],
        }

    except Exception as e:
        logger.error(f"Error fetching Google Trends data: {e}")
        return {"status": "error", "error": str(e)}


async def fetch_price_history(query: str, api_key: str) -> Dict[str, Any]:
    from serpapi import GoogleSearch
    results_by_period: Dict[str, Dict] = {}

    for timeframe, label in [("now 1-d", "24h"), ("now 7-d", "7gg"), ("today 3-m", "3mesi")]:
        params = {
            "engine": "google_trends",
            "q": query,
            "data_type": "TIMESERIES",
            "date": timeframe,
            "api_key": api_key,
        }
        try:
            def _search(p=params):
                return GoogleSearch(p).get_dict()

            data = await asyncio.to_thread(_search)
            timeline = data.get("interest_over_time", {}).get("timeline_data", [])
            values: List[int] = []
            for item in timeline:
                for v in item.get("values", []):
                    try:
                        values.append(int(v.get("extracted_value", 0)))
                    except ValueError:
                        pass

            results_by_period[label] = {
                "avg": round(sum(values) / len(values), 1) if values else 0,
                "peak": max(values) if values else 0,
            }
        except Exception as e:
            logger.error(f"Error fetching price history [{label}]: {e}")
            results_by_period[label] = {"avg": 0, "peak": 0}

    return {"status": "ok", "periods": results_by_period}


def build_verdetto(shopping_data: Dict, trends_data: Dict) -> str:
    if shopping_data.get("status") != "ok":
        return "Analisi non disponibile."

    avg = shopping_data["average_price"]
    min_p = shopping_data["min_price"]
    max_p = shopping_data["max_price"]
    std_dev = shopping_data.get("std_dev", 0)
    price_consistency = shopping_data.get("price_consistency", "")

    spread = max_p - min_p
    if spread > 0 and avg < (min_p + spread * 0.3):
        base = f"OTTIMA OPPORTUNITÀ. Il prezzo medio (€{avg}) è vicino al minimo riscontrato (€{min_p})."
    elif spread > 0 and avg > (min_p + spread * 0.7):
        base = f"PREZZO SOPRA LA MEDIA. Al momento il mercato propone prezzi alti (€{avg}). Valuta di attendere."
    else:
        base = f"PREZZO EQUO. Il valore di €{avg} è in linea con il mercato attuale."

    if trends_data.get("status") == "ok":
        trend = trends_data["trend_direction"]
        interest = trends_data["current_interest"]
        trend_labels = {
            "forte crescita": f"Ottima offera ({interest}/100). {base} La domanda è in forte aumento: agisci ora prima che i prezzi salgano.",
            "in crescita":    f"Non Fartela sfuggire ({interest}/100). {base} L'interesse è in aumento, i prezzi potrebbero salire a breve.",
            "in calo":        f" DOMANDA IN CALO. {base} L'interesse sta diminuendo: potresti trovare sconti a breve.",
            "forte calo":     f" DOMANDA IN FORTE CALO. {base} Il mercato si sta raffreddando rapidamente: aspetta ancora per prezzi migliori.",
            "stabile":        f" MERCATO STABILE. {base} L'interesse è costante ({interest}/100).",
        }
        verdetto = trend_labels.get(trend, f"MERCATO STABILE. {base}")
    else:
        verdetto = base + " (Dati sui trend non disponibili per questa ricerca)."

    if price_consistency:
        verdetto += f" Nota: {price_consistency} (σ=€{std_dev})."

    return verdetto


async def get_market_trends_analysis(query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Funzione orchestrata per ottenere l'analisi completa dei trend.
    """
    key = api_key or SERP_API_KEY
    if not key:
        return {"status": "error", "message": "Manca SERP_API_KEY"}

    tasks = [
        fetch_google_shopping(query, key),
        fetch_google_trends(query, key),
        fetch_price_history(query, key)
    ]
    shopping_data, trends_data, history_data = await asyncio.gather(*tasks)
    verdetto = build_verdetto(shopping_data, trends_data)

    status = "ok"
    if shopping_data.get("status") == "error" and trends_data.get("status") == "error":
        status = "error"

    return {
        "status": status,
        "query": query,
        "shopping_data": shopping_data,
        "trends_data": trends_data,
        "history_data": history_data,
        "verdetto": verdetto,
    }
