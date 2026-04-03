import sys
import os
import time
import logging

# Aggiungi la root del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.browser_manager import BrowserManager

# Configura logging per vedere cosa succede
logging.basicConfig(level=logging.INFO)

def test_navigation():
    mgr = BrowserManager.get_instance()
    mgr.set_timeout(30) # 30 secondi per il test
    
    print("--- 1. Navigating to google.it ---")
    res = mgr.navigate("https://www.google.it")
    print(f"Status: {res['status']} | URL: {res['url']} | Title: {res['title']}")
    
    if res['screenshot']:
        print("Screenshot captured (base64 length:", len(res['screenshot']), ")")
    
    print("\n--- 2. Getting view (persistence check) ---")
    res2 = mgr.get_view()
    print(f"Status: {res2['status']} | URL: {res2['url']}")
    
    print("\n--- 3. Closing browser ---")
    mgr.close()
    print("Done.")

if __name__ == "__main__":
    test_navigation()
