"""
test_playwright_tool.py
Test standalone per il servizio Playwright eBay scraper.
Esegui dalla root del progetto con la venv attivata:
    python app/test/test_playwright_tool.py
oppure con pytest:
    python -m pytest app/test/test_playwright_tool.py -v
"""

import asyncio
import sys
import os

# Assicura che 'app' sia importabile dalla root del progetto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def _run_test():
    from app.services.ebay_playwright import scrape_ebay_search

    print("\n--- Test: scrape_ebay_search('iphone 13', max_results=5) ---")
    results = await scrape_ebay_search("iphone 13", max_results=5)

    assert isinstance(results, list), "Il risultato deve essere una lista"
    assert len(results) > 0, "Deve esserci almeno un risultato"

    for i, item in enumerate(results):
        assert "title" in item and item["title"], f"Item {i} mancante di title"
        assert "url" in item and item["url"], f"Item {i} mancante di url"
        print(f"  [{i+1}] {item['title'][:60]} | {item.get('price_raw', 'N/A')} | {item['url'][:50]}...")

    print(f"\n✅ Test PASSED — {len(results)} risultati trovati.\n")


def test_scrape_ebay_search():
    """Wrapper pytest-compatible."""
    asyncio.run(_run_test())


if __name__ == "__main__":
    asyncio.run(_run_test())
