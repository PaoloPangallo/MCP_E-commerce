import sys
import os
import asyncio
import json
import base64

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.mcp.tools.browser_tools import browser_navigate, browser_type, browser_click, browser_get_view, browser_close

async def run_integration_test():
    print("=== STARTING MCP BROWSER INTEGRATION TEST ===")
    
    # 1. Navigate
    print("\n[Step 1] Navigating to eBay...")
    res_nav = await browser_navigate(url="https://www.ebay.it")
    print(f"Status: {res_nav.get('status')} | URL: {res_nav.get('url')} | Title: {res_nav.get('title')}")
    
    # 2. Type search query
    # eBay search input is usually #gh-ac or .gh-tb
    print("\n[Step 2] Typing 'macbook' in search bar...")
    res_type = await browser_type(selector="#gh-ac", text="macbook")
    print(f"Status: {res_type.get('status')} | URL: {res_type.get('url')}")
    
    # 3. Click search button
    # eBay search button is usually #gh-btn
    print("\n[Step 3] Clicking search button...")
    res_click = await browser_click(selector="#gh-btn")
    print(f"Status: {res_click.get('status')} | URL: {res_click.get('url')} | Title: {res_click.get('title')}")
    
    # 4. Get View and verify
    print("\n[Step 4] Verifying final state...")
    res_view = await browser_get_view()
    final_url = res_view.get('url', '')
    print(f"Final URL: {final_url}")
    
    if "sch/i.html" in final_url or "macbook" in final_url.lower():
        print("SUCCESS: We are on the search results page!")
    else:
        print("WARNING: URL doesn't look like a search result page.")
        
    if res_view.get('screenshot'):
        # Save a small snippet to prove it exists
        img_data = res_view.get('screenshot')
        print(f"Screenshot received (length: {len(img_data)})")
        with open("tmp/test_screenshot.jpg", "wb") as f:
            f.write(base64.b64decode(img_data))
        print("Screenshot saved to tmp/test_screenshot.jpg")

    # 5. Close
    print("\n[Step 5] Closing browser...")
    await browser_close()
    print("Test completed.")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
