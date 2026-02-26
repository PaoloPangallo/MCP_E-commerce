import spacy
import re

nlp = spacy.load("it_core_news_sm")

def extract_max_price(text: str):
    match = re.search(r"sotto\s+(\d+)", text.lower())
    if match:
        return float(match.group(1))
    return None

def extract_condition(doc):
    for token in doc:
        if token.text.lower() in ["usato", "nuovo", "ricondizionato"]:
            return token.text.lower()
    return None

def extract_product(doc):
    stop_words = {"euro", "eur"}
    product_tokens = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            if token.text.lower() not in stop_words:
                product_tokens.append(token.text)

    return " ".join(product_tokens) if product_tokens else None

def parse_query_service(text: str):
    doc = nlp(text)

    return {
        "original_query": text,
        "product": extract_product(doc),
        "max_price": extract_max_price(text),
        "condition": extract_condition(doc),
    }
