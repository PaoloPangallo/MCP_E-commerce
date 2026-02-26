"""
test_ebay_connection.py

Test semplice per:
- Ottenere OAuth token (Sandbox)
- Fare una search di prova con Browse API
"""

import os
import base64
import requests
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

OAUTH_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"

MARKETPLACE_ID = "EBAY_IT"


# ============================================================
# STEP 1 — GET OAUTH TOKEN
# ============================================================

def get_oauth_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("EBAY_CLIENT_ID o EBAY_CLIENT_SECRET non trovati nel .env")

    print("🔐 Richiesta OAuth token...")

    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_auth}",
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    response = requests.post(OAUTH_URL, headers=headers, data=data)

    if response.status_code != 200:
        print("❌ Errore OAuth:")
        print(response.status_code, response.text)
        raise RuntimeError("OAuth fallito")

    token_data = response.json()
    print("✅ OAuth OK")
    print("⏳ Token valido per:", token_data.get("expires_in"), "secondi")

    return token_data["access_token"]


# ============================================================
# STEP 2 — TEST SEARCH
# ============================================================

def test_search(token):
    print("\n🔎 Test ricerca eBay Sandbox...")

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID,
    }

    params = {
        "q": "iphone",
        "limit": 5
    }

    response = requests.get(SEARCH_URL, headers=headers, params=params)

    if response.status_code != 200:
        print("❌ Errore Search:")
        print(response.status_code, response.text)
        raise RuntimeError("Search fallita")

    data = response.json()

    items = data.get("itemSummaries", [])

    print(f"✅ Search OK - trovati {len(items)} risultati\n")

    for i, item in enumerate(items, 1):
        title = item.get("title")
        price_info = item.get("price", {})
        price = price_info.get("value")
        currency = price_info.get("currency")
        print(f"{i}. {title}")
        print(f"   💰 {price} {currency}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("===================================")
    print(" TEST CONNESSIONE EBAY SANDBOX ")
    print("===================================\n")

    token = get_oauth_token()
    test_search(token)

    print("🎉 Test completato con successo.")