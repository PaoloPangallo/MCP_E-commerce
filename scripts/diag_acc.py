import re
import numpy as np

ACCESSORY_WORDS = [
    "case", "cover", "charger", "caricatore", "cavo", "cable", "vetro", "pellicola",
    "screen protector", "glass", "custodia", "adattatore", "adapter", "scocca",
    "ricambio", "parte", "batteria", "battery", "display", "schermo", "vetrino",
    "bumper", "stand", "holder", "supporto", "alimentatore", "power", "jack",
    "keyboard", "tastiera", "mouse", "cuffie", "headphones", "earbuds", "auricolari",
    "box", "scatola", "confezione", "manuale", "manual", "dummy", "finto", "mostra",
    "pezzi", "spare", "repair", "flat", "sensore", "sensor", "connettore", "connector",
    "flex", "modulo", "tasto", "scocca", "lente", "camera", "ricarica"
]

def accessory_penalty(query: str, title: str, product_type: str = "Unknown") -> float:
    q = (query or "").lower()
    t = (title or "").lower()

    if product_type in ["Accessory", "Part"]:
        return 0.0

    is_main_intent = product_type in ["Main", "Unknown"]
    
    # Use re for robust tokenization
    t_words = set(re.findall(r"\w+", t))
    q_words = set(re.findall(r"\w+", q))
    
    acc_in_title = any(aw in t for aw in ACCESSORY_WORDS)
    acc_in_query = any(aw in q for aw in ACCESSORY_WORDS)

    print(f"Title: {t}")
    print(f"Query: {q}")
    print(f"Acc in Title: {acc_in_title}")
    print(f"Acc in Query: {acc_in_query}")

    if is_main_intent and acc_in_title and not acc_in_query:
        return 2.0
        
    return 0.0

# TEST CASE from screenshot
query = "lo cerchiamo un iphone 12?"
title = "Batteria nuova per APPLE iPhone 12 13 14 15 Pro Max NO errore e..."
res = accessory_penalty(query, title, "Main")
print(f"RESULT: {res}")
