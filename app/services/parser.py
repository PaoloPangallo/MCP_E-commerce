import spacy
import re
import subprocess
import json

nlp = spacy.load("it_core_news_sm")

# ==========================================
# LLM CALL
# ==========================================

def call_llm(prompt: str):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            timeout=20
        )
        return result.stdout.strip()
    except Exception as e:
        print("LLM error:", e)
        return None


def llm_parse(query: str):
    prompt = f"""
Return ONLY valid JSON.

Schema:
{{
  "brands": list of strings,
  "product": string or null,
  "base_constraints": {{
      "min_price": float or null,
      "max_price": float or null,
      "condition": string or null
  }},
  "conditional_constraints": [
      {{
        "if_condition": string,
        "max_price": float
      }}
  ],
  "preferences": {{
      "preferred_condition": string or null,
      "acceptable_conditions": list of strings
  }}
}}

Extract structured information from:

"{query}"
"""

    response = call_llm(prompt)

    if not response:
        return None

    try:
        return json.loads(response)
    except:
        print("Invalid JSON:", response)
        return None


# ==========================================
# RULE-BASED BASELINE
# ==========================================

def normalize(text):
    return text.lower().replace("€", " euro ")


def extract_brands(doc):
    brands = []
    for token in doc:
        if token.pos_ == "PROPN":
            brands.append(token.text.capitalize())
    return list(set(brands))


def extract_base_price(text):
    text = normalize(text)

    match = re.search(r"(sui|intorno a|circa)\s+(\d+)", text)
    if match:
        return None, float(match.group(2))

    match = re.search(r"(tra|da)\s+(\d+)\s+(e|a)\s+(\d+)", text)
    if match:
        return float(match.group(2)), float(match.group(4))

    return None, None


def parse_query_service(text: str):
    normalized = normalize(text)
    doc = nlp(normalized)

    # RULE-BASED MINIMALE
    brands = extract_brands(doc)
    min_price, max_price = extract_base_price(normalized)

    rule_result = {
        "original_query": text,
        "brands": brands,
        "product": None,
        "base_constraints": {
            "min_price": min_price,
            "max_price": max_price,
            "condition": None
        },
        "conditional_constraints": [],
        "preferences": {
            "preferred_condition": None,
            "acceptable_conditions": []
        }
    }

    # Sempre raffinamento LLM (schema avanzato)
    llm_result = llm_parse(text)

    if llm_result:
        llm_result["original_query"] = text
        return llm_result

    return rule_result