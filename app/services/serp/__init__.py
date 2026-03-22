from .ebay_deals import fetch_ebay_deals, DealsResult, DealItem
from .market_trends import get_market_trends_analysis, fetch_google_shopping, fetch_google_trends, fetch_price_history

__all__ = [
    "fetch_ebay_deals",
    "DealsResult",
    "DealItem",
    "get_market_trends_analysis",
    "fetch_google_shopping",
    "fetch_google_trends",
    "fetch_price_history"
]
