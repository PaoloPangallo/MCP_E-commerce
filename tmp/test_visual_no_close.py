import sys
import os
import asyncio
import json
import base64
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.mcp.tools.browser_tools import browser_navigate, browser_type, browser_click, browser_get_view

async def run_visual_test():
    print("=== LIVE VISUAL TEST ===")
    
    # 1. Navigate
    print("\n[Step 1] Navigating to eBay...")
    await browser_navigate(url="https://www.ebay.it")
    
    # Wait for user to see
    await asyncio.sleep(2)
    
    print("\n[Step 1.5] Accept Cookies if present...")
    await browser_click(selector="#gdpr-banner-accept")

    
    # 2. Type search query
    print("\n[Step 2] Typing 'macbook'...")
    # Using #gh-ac for input
    await browser_type(selector="#gh-ac", text="macbook")
    
    await asyncio.sleep(1)
    
    # 3. Click search button
    # Trying both common selectors for reliability in test
    print("\n[Step 3] Clicking search button...")
    # On eBay.it the button is often <input type="submit" id="gh-btn">
    # The subagent suggested #gh-search-btn, let's try to find it first.
    res = await browser_click(selector="#gh-search-btn")
    
    print(f"Click Result URL: {res.get('url')}")
    
    # 4. Wait for results to be visible
    print("\n[Step 4] Waiting 5 seconds for results to load...")
    await asyncio.sleep(5)
    
    res_final = await browser_get_view()
    print(f"Final URL: {res_final.get('url')}")
    print(f"Final Title: {res_final.get('title')}")
    
    print("\nTest completed. I will NOT close the browser so you can check the window.")

if __name__ == "__main__":
    asyncio.run(run_visual_test())
