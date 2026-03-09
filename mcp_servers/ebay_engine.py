import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# We import the existing functional core for now to wrap it into MCP tools
from app.services.ebay import search_items, get_category_aspects

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("eBayEngineServer")

class SearchFilters(BaseModel):
    min_price: Optional[float] = Field(None, description="Minimum price constraint in EUR")
    max_price: Optional[float] = Field(None, description="Maximum price constraint in EUR")
    preferences: List[Dict[str, Any]] = Field(default_factory=list, description="User preferences like condition (new, used, etc.)")

class SearchProductsInput(BaseModel):
    semantic_query: str = Field(..., description="The highly optimized search term (e.g., 'Levi's 501 neri uomo')")
    filters: Optional[SearchFilters] = Field(None, description="Filters to apply to the search")
    limit: int = Field(5, description="Number of products to return")

@mcp.tool()
def search_products(semantic_query: str, min_price: Optional[float] = None, max_price: Optional[float] = None, condition: Optional[str] = None, limit: int = 5, allow_relaxed: bool = False) -> str:
    """
    Search for products on eBay using the provided semantic query and optional filters.
    
    Args:
        semantic_query: Optimized search term (e.g., "Levi's 501 black L")
        min_price: Minimum item price in EUR
        max_price: Maximum item price in EUR
        condition: Item condition, e.g., 'new', 'used'
        limit: Number of items to retrieve
        allow_relaxed: If True, constraints like price/condition will be ignored if 0 exact results are found, showing "close" matches.
    """
    logger.info(f"Executing search_products with query='{semantic_query}'")
    
    # Reconstruct the constraints structure used by the underlying service
    constraints = []
    if min_price is not None or max_price is not None:
        price_constraint = {"type": "price", "operator": "between"}
        val = []
        if min_price is not None:
            val.append(min_price)
        else:
            val.append(0.0)
        
        if max_price is not None:
            val.append(max_price)
        else:
            val.append(999999.0) # Arbitrary large max
            
        price_constraint["value"] = val
        constraints.append(price_constraint)
        
    if condition:
        constraints.append({"type": "condition", "value": condition})
        
    parsed_input = {
        "semantic_query": semantic_query,
        "brands": [], # Will be handled by the semantic_query internally
        "product": semantic_query,
        "constraints": constraints,
        "preferences": [],
        "compatibilities": {}
    }
    
    try:
        results = search_items(parsed_query=parsed_input, limit=limit, ignore_constraints=allow_relaxed)
        # Convert results list to json string. In case search_items returns something else, ensure it is handleable.
        return json.dumps(results if results else [], ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return json.dumps({"error": str(e), "results": []})

@mcp.tool()
def get_category_metadata(category_id: str) -> str:
    """
    Get the appropriate eBay category aspects for a given category id.
    
    Args:
        category_id: A string representing what the user is searching for (e.g., '101')
    """
    logger.info(f"Executing get_category_metadata with category_id={category_id}")
    try:
        aspects = get_category_aspects(category_id)
        return json.dumps({
            "category_id": category_id,
            "required_aspects": aspects
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error finding category aspects: {e}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Start the server using stdio transport
    mcp.run(transport='stdio')
