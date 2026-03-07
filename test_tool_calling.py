"""
Test diagnostico VELOCE con il formato tool results aggiornato.
"""
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral-nemo")

import sys
sys.path.insert(0, ".")
from app.services.agent_tools import AGENT_TOOLS_SCHEMA

SYS_PROMPT = (
    "You are an eBay shopping assistant. You MUST use tools to help the user. "
    "NEVER answer directly — always use the tools.\n\n"
    "MANDATORY FLOW:\n"
    "1. Call parse_query with the user's message.\n"
    "2. Immediately call search_products with the parsed query. Do NOT write any text.\n"
    "3. Call explain_results to present findings to the user.\n\n"
    "IMPORTANT: After receiving a tool result, call the NEXT tool. "
    "Do NOT write text between tool calls. Only speak after explain_results.\n"
    "If the user's request is vague (missing size, color, storage), "
    "call request_user_clarification INSTEAD of parse_query."
)

def test(name, messages):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    resp = requests.post(OLLAMA_CHAT_URL, json={
        "model": MODEL_NAME,
        "messages": messages,
        "tools": AGENT_TOOLS_SCHEMA,
        "stream": False,
    }, timeout=120)
    
    msg = resp.json().get("message", {})
    tc = msg.get("tool_calls", [])
    content = msg.get("content", "").strip()
    
    if tc:
        for t in tc:
            fn = t["function"]["name"]
            args = t["function"]["arguments"]
            print(f"  ✅ TOOL CALL: {fn}({json.dumps(args, ensure_ascii=False)})")
    else:
        print(f"  ❌ TEXT: '{content[:200]}'")
    
    return msg

# ============================================================
# TEST 1: User -> parse_query  
# ============================================================
msg1 = test("STEP 1: user msg → parse_query?", [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "cerco scarpe nike sotto i 150 euro"},
])

# ============================================================
# TEST 2: dopo parse_query -> search_products
# NUOVO: risultato semplificato con next_action
# ============================================================
parse_result = json.dumps({
    "status": "parsed",
    "search_query": "scarpe nike",
    "next_action": "NOW call search_products with query: scarpe nike"
})

msg2 = test("STEP 2: parse_query result → search_products?", [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "cerco scarpe nike sotto i 150 euro"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "parse_query", "arguments": {"query": "cerco scarpe nike sotto i 150 euro"}}}
    ]},
    {"role": "tool", "name": "parse_query", "content": parse_result},
])

# ============================================================  
# TEST 3: dopo search_products -> explain_results
# NUOVO: risultato semplificato con next_action
# ============================================================
search_result = json.dumps({
    "status": "search_complete",
    "results_count": 20,
    "top_results": [
        {"title": "Nike Air Max 95", "price": 89.99, "trust": 0.82},
        {"title": "Nike Shox R4", "price": 78.0, "trust": 0.85},
    ],
    "next_action": "NOW call explain_results with query: scarpe nike"
})

msg3 = test("STEP 3: search result → explain_results?", [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "cerco scarpe nike sotto i 150 euro"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "parse_query", "arguments": {"query": "cerco scarpe nike sotto i 150 euro"}}}
    ]},
    {"role": "tool", "name": "parse_query", "content": parse_result},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "search_products", "arguments": {"query": "scarpe nike"}}}
    ]},
    {"role": "tool", "name": "search_products", "content": search_result},
])

print(f"\n{'='*60}")
print("  RIEPILOGO")
print(f"{'='*60}")
tc1 = msg1.get("tool_calls", [])
tc2 = msg2.get("tool_calls", [])
tc3 = msg3.get("tool_calls", [])
print(f"  Step 1 (→parse_query):      {'✅' if tc1 and tc1[0]['function']['name']=='parse_query' else '❌'}")
print(f"  Step 2 (→search_products):  {'✅' if tc2 and tc2[0]['function']['name']=='search_products' else '❌'}")
print(f"  Step 3 (→explain_results):  {'✅' if tc3 and tc3[0]['function']['name']=='explain_results' else '❌'}")
