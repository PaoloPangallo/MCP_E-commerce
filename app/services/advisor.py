import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

async def get_market_insights(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes if the item price is a good deal compared to market data.
    Uses the eBay Product ID (epid) if available.
    """
    title = item.get("title", "Prodotto")
    price = item.get("price", 0)
    epid = item.get("epid")
    
    logger.info("Advisor: Analyzing market for '%s' (price: %s, epid: %s)", title, price, epid)
    
    # Mocking historical data based on epid/title
    # In a real scenario, we would call:
    # https://api.ebay.com/commerce/catalog/v1/product_summary/search?q={title}
    # or get product details by epid.
    
    # Simulate historical average price
    # (Using hash of title/epid for stability in the same session)
    seed = sum(ord(c) for c in (epid or title))
    random.seed(seed)
    
    avg_market_price = price * random.uniform(0.8, 1.25)
    
    is_deal = price < (avg_market_price * 0.9)
    is_expensive = price > (avg_market_price * 1.1)
    
    if is_deal:
        verdict = "GREAT_DEAL"
        advice = "Questo prezzo è sotto la media di mercato! È un ottimo affare."
    elif is_expensive:
        verdict = "OVERPRICED"
        advice = "Il prezzo sembra superiore alla media storica per questo modello."
    else:
        verdict = "FAIR_PRICE"
        advice = "Il prezzo è in linea con le quotazioni di mercato."
        
    return {
        "verdict": verdict,
        "average_market_price": round(avg_market_price, 2),
        "advice": advice,
        "currency": item.get("currency", "EUR"),
        "confidence": 0.85 if epid else 0.6
    }
