import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.services.parser import correct_brands_in_text, parse_query_service

def test_query_corruption():
    query = "quando mi arriverebbe iphone 14 con cap 89030"
    
    print(f"Original Query: {query}")
    
    # Test brand correction
    corrected = correct_brands_in_text(query)
    print(f"Corrected Text (Expected 'iphone' kept, '14' and '89030' kept): {corrected}")
    
    # Test full parsing
    parsed = parse_query_service(query, use_llm=False) # Test rule-based first
    print(f"Parsed Brands: {parsed.get('brands')}")
    print(f"Parsed Product: {parsed.get('product')}")
    
    # Assertions
    assert "iPhone" in [b for b in parsed.get('brands')], f"Expected iPhone in brands, got {parsed.get('brands')}"
    assert "Epiphone" not in corrected, "Epiphone should not be in corrected text"
    assert "Bando" not in corrected, "Bando should not be in corrected text"
    
    print("\nSUCCESS: No more query corruption for 'iphone 14'!")

if __name__ == "__main__":
    try:
        test_query_corruption()
    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)
