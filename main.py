from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import re

# =========================
# Setup
# =========================

app = FastAPI(title="MCP E-Commerce NLP Backend")

# Carica modello spaCy italiano
nlp = spacy.load("it_core_news_sm")


# =========================
# Request Model
# =========================

class QueryRequest(BaseModel):
    query: str


# =========================
# Utility Parsing
# =========================

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
    # Semplice: prendi sostantivi principali
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return " ".join(nouns) if nouns else None


# =========================
# Endpoints
# =========================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/parse")
def parse_query(request: QueryRequest):
    text = request.query
    doc = nlp(text)

    structured_query = {
        "original_query": text,
        "product": extract_product(doc),
        "max_price": extract_max_price(text),
        "condition": extract_condition(doc),
    }

    return structured_query