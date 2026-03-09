import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

from app.services.parser import llm_parse, deep_refine_parse
from app.services.trust import compute_trust_score
# Import ranking logic if available (agent_tools.process_and_rank or similar)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("AiAnalystServer")

@mcp.tool()
def parse_intent(query: str, previous_context: Optional[str] = None) -> str:
    """
    Parse a natural language query into a structured semantic query, identifying product, brand, attributes, and missing info.
    
    Args:
        query: The user's prompt (e.g. 'cerca scarpe nike blu 42')
        previous_context: Optional JSON string of previous state to maintain conversations (e.g., '{"compatibilities": {"Size": "42"}}')
    """
    context_dict = {}
    if previous_context:
        try:
            context_dict = json.loads(previous_context)
        except:
            pass
            
    try:
        parsed_result, _ = llm_parse(query, previous_context=context_dict)
        if not parsed_result:
            return json.dumps({"error": "Failed to parse intent", "semantic_query": query})
        
        # Second pass optional refinement if we know the category
        # But for simple MCP tool we return the base parsed result
        return json.dumps(parsed_result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error parsing intent: {e}")
        return json.dumps({"error": str(e), "semantic_query": query})

@mcp.tool()
def calculate_trust(seller_stats: dict) -> str:
    """
    Calculate a trust score for a seller given their statistics.
    
    Args:
        seller_stats: JSON object with seller metrics (feedback_score, positive_feedback_percent, seller_level, top_rated_seller)
    """
    try:
        feedback_score = seller_stats.get("feedback_score", 0)
        positive_feedback_percent = seller_stats.get("positive_feedback_percent", 0.0)
        seller_level = seller_stats.get("seller_level", "")
        top_rated = seller_stats.get("top_rated_seller", False)
        
        # Dummy structure simulating expected trust score inputs from original logic
        # We need to construct dummy feedbacks list
        dummy_feedbacks = []
        positive_n = int(positive_feedback_percent)
        negative_n = 100 - positive_n
        for _ in range(positive_n):
            dummy_feedbacks.append({"rating": "Positive", "time": "2024-01-01T00:00:00Z"})
        for _ in range(negative_n):
            dummy_feedbacks.append({"rating": "Negative", "time": "2024-01-01T00:00:00Z"})
            
        score = compute_trust_score(
            feedbacks=dummy_feedbacks,
            sentiment_score=positive_feedback_percent / 100.0
        )
        return json.dumps({"trust_score": score}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error calculating trust: {e}")
        return json.dumps({"error": str(e), "trust_score": 0.0})

@mcp.tool()
def rank_results(products: List[dict], user_preferences: Optional[str] = None) -> str:
    """
    Rank and filter a list of products using LLM-based intelligence and user preferences.
    
    Args:
        products: JSON list of products from eBay
        user_preferences: Optional user prompt explaining what they care about most (e.g., 'condition, price')
    """
    # Dummy mock for ranking logic - you would import your actual ranking function here
    # For now we just return the items sorted by trust_score if it exists, or price
    # True implementation would call the local LLM to rank based on semantic fit
    
    # Sort purely by price as an example, since LLM ranking requires the full prompt
    sorted_products = sorted(products, key=lambda x: (x.get("price", {}).get("value", 9999)))
    return json.dumps({"ranked_results": sorted_products[:5]}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport='stdio')
