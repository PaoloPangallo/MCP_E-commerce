import sys
import os
from pathlib import Path

# Aggiungi la root del progetto al path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from app.services.parser import rule_based_parse, parse_query_service
from app.services.ebay import _build_query

def test_query(query_text):
    print(f"\n--- TESTING QUERY: {query_text} ---")
    
    from app.services.parser import get_nlp
    doc = get_nlp()(query_text)
    print("Tokens:", [(t.text, t.pos_, t.is_stop) for t in doc])
    print("Noun Chunks:", [chunk.text for chunk in doc.noun_chunks])

    # 1. Test Rule-based
    parsed_rules = rule_based_parse(query_text)
    print(f"Rule-based parsed: product='{parsed_rules.get('product')}', brands={parsed_rules.get('brands')}, constraints={parsed_rules.get('constraints')}")
    
    ebay_q_rules = _build_query(parsed_rules)
    print(f"Rule-based eBay Query: {ebay_q_rules}")

    # 2. Se possibile, test LLM (simulato se non abbiamo connessione, o saltato)
    # parsed_llm = parse_query_service(query_text, use_llm=True)
    # print(f"LLM-based parsed: {parsed_llm}")
    # ebay_q_llm = _build_query(parsed_llm)
    # print(f"LLM-based eBay Query: {ebay_q_llm}")

if __name__ == "__main__":
    test_query("iphone con 512 gb di storage")
    test_query("Samsung Galaxy S22 usato")
    test_query("iphone 15 pro 128gb")
