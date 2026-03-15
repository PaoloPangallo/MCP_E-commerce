import logging
import json
from app.services.parser import parse_query_service

# Setup logging to see the Ollama interaction
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.services.parser")
logger.setLevel(logging.DEBUG)

def test_parser():
    query = "cerco un iphone 13 sotto i 500 euro"
    print(f"Testing query: {query}")
    
    # This should trigger LLM because it's complex enough
    result = parse_query_service(query, use_llm=True)
    
    print("\nPARSER RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("_meta", {}).get("llm_success"):
        print("\n✅ LLM Success!")
    else:
        print("\n❌ LLM Failed (check logs for 404 or other errors)")

if __name__ == "__main__":
    test_parser()
