import requests
from bs4 import BeautifulSoup
import json

URL = "https://www.ebay.it/n/all-brands"

def fetch_brands():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")

    brands = []

    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        if text and len(text) > 1 and text.isalnum():
            brands.append(text)

    brands = sorted(set(brands))

    with open("brand_vocab.json", "w") as f:
        json.dump(brands, f, indent=2)

    print(f"Saved {len(brands)} brands")

fetch_brands()