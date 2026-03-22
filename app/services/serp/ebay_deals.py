import os
import logging
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SERP_API_KEY = os.getenv("SERP_API_KEY")
SERP_BASE_URL = "https://serpapi.com/search.json"

class DealItem(BaseModel):
    title: str
    link: str
    price: Dict[str, Any]
    old_price: Optional[Dict[str, Any]] = None
    thumbnail: Optional[str] = None
    product_id: Optional[str] = None
    extensions: List[str] = []

class DealsResult(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    items: List[DealItem] = []

async def fetch_ebay_deals(
    query: Optional[str] = None,
    category_id: Optional[str] = None,
    ebay_domain: str = "ebay.it"
) -> Optional[DealsResult]:
    """
    Fetches deals from eBay using SerpApi.
    """
    if not SERP_API_KEY:
        logger.error("SERP_API_KEY not found in environment")
        # Consideriamo la possibilità di sollevare un'eccezione che il tool MCP catturerà
        raise ValueError("Configurazione mancante: SERP_API_KEY non trovata nell'ambiente del server.")

    params = {
        "engine": "ebay",
        "api_key": SERP_API_KEY,
        "ebay_domain": ebay_domain,
    }

    if query:
        params["_nkw"] = query # eBay search query
    
    # Se mancano sia query che category_id, usiamo una categoria di default (es: 9355 - Cellulari) 
    # per evitare il 400 Bad Request di SerpApi dell'engine ebay.
    effective_category = category_id
    if not query and not category_id:
        effective_category = "9355" # Default to Consumer Electronics / Cell Phones common category
        logger.info("No query or category provided for eBay deals. Using default category 9355.")

    if effective_category:
        params["category_id"] = effective_category

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(SERP_BASE_URL, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            deals_data = data.get("deals", {})
            if not deals_data or not deals_data.get("items"):
                # If no direct deals section, try to look into results if any? 
                # serpapi usually has a "deals" key for specific deal engines
                logger.info("No 'deals' section found in SerpApi eBay response")
                return None

            items = []
            for item in deals_data.get("items", []):
                items.append(DealItem(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    price=item.get("price", {}),
                    old_price=item.get("old_price"),
                    thumbnail=item.get("thumbnail"),
                    product_id=item.get("product_id"),
                    extensions=item.get("extensions", [])
                ))

            return DealsResult(
                title=deals_data.get("title"),
                subtitle=deals_data.get("subtitle"),
                items=items
            )

        except Exception as e:
            logger.error(f"Error fetching deals from SerpApi: {e}")
            return None
