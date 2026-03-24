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
    
    # Se mancano sia query che category_id, usiamo una categoria di default
    effective_category = category_id
    effective_query = query
    if not query and not category_id:
        effective_category = "9355"
        logger.info("No query or category provided for eBay deals. Using default category 9355.")
    
    # Se abbiamo una categoria ma non una query, dobbiamo fornire una query a SerpApi per evitare 400 Bad Request
    if effective_category and not effective_query:
        effective_query = "offerte"

    if effective_query:
        params["_nkw"] = effective_query
    if effective_category:
        params["_sacat"] = effective_category

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(SERP_BASE_URL, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            deals_data = data.get("deals", {})
            items = []
            
            # 1. Prova a estrarre deals "ufficiali" dalla pagina Deals
            if deals_data and deals_data.get("items"):
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
                    title=deals_data.get("title", "Offerte in Evidenza"),
                    subtitle=deals_data.get("subtitle", ""),
                    items=items
                )
            
            # 2. Fallback strategico: usa organic_results se abbiamo cercato in una categoria specifica
            organic = data.get("organic_results", [])
            if organic:
                logger.info("No explicit 'deals' section found, falling back to top organic results as deals.")
                for item in organic[:20]:
                    items.append(DealItem(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        price=item.get("price", {}),
                        old_price=item.get("old_price"),
                        thumbnail=item.get("thumbnail"),
                        product_id=None,
                        extensions=item.get("extensions", [])
                    ))
                
                cat_name = effective_query if effective_query != "offerte" else "Selezionata"
                return DealsResult(
                    title=f"Migliori Offerte (Categoria {cat_name})",
                    subtitle="Risultati top per la ricerca",
                    items=items
                )
                
            logger.info("No 'deals' or 'organic_results' found in SerpApi eBay response")
            return None

        except Exception as e:
            logger.error(f"Error fetching deals from SerpApi: {e}")
            return None
