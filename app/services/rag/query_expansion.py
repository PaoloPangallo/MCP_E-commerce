import logging
from typing import Optional
from app.services.parser import call_llm

logger = logging.getLogger(__name__)

async def expand_query(query: str, llm_engine: Optional[str] = None) -> str:
    """
    Expands the user query using a fast LLM call.
    Example: 'compara iphone 15 max' -> 'Apple iPhone 15 Pro Max smartphone comparison'
    This helps the Sparse (BM25) and Dense retrieval stages find more relevant matches.
    """
    if not query or not query.strip():
        return ""
        
    prompt = f"""
    You are an AI search assistant for an e-commerce platform.
    Your task is to expand the user's short query into a slightly more descriptive, keyword-rich query.
    Add relevant brand names, categories, or synonyms if applicable.
    Keep it under 10 words. Do NOT include phrases like 'here is the expansion' or quotes.
    Just return the expanded query text.

    User query: "{query}"
    """
    try:
        # Use generic call_llm to respect preferred provider/engine
        res, used_provider = await call_llm(prompt, engine=llm_engine)
        if res is None:
            return query
            
        expanded = res.strip().replace('"', '').replace('\n', ' ')
        
        # Fallback if the LLM returns something weird or too long
        if len(expanded.split()) > 20 or not expanded:
            return query
            
        return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query
