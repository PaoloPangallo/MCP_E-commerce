import json
import time
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, ".")

# Mock DB and User for testing
class MockSession:
    def query(self, *args, **kwargs): return self
    def filter_by(self, *args, **kwargs): return self
    def first(self): return None
    def add(self, item): pass
    def commit(self): pass
    def rollback(self): pass

from app.services.agent_graph import run_agent_graph

def test_scenario(query: str):
    print(f"\n🚀 TESTING SCENARIO: {query}")
    
    shared_state = {
        "results": [],
        "parsed_query": None,
        "thinking_trace": [],
        "rag_context": "",
        "metrics": {},
        "_timings": {},
        "seller_feedbacks": {},
        "db_session": MockSession(),
        "user_obj": None,
    }
    
    t0 = time.time()
    result = run_agent_graph(
        user_message=query,
        history=[],
        shared_state=shared_state
    )
    
    print("\n--- FINAL AGENT RESPONSE ---")
    print(result.get("analysis"))
    print("\n--- THINKING TRACE ---")
    for trace in result.get("thinking_trace", []):
        print(f"DEBUG: {trace}")
    
    print("\n--- RESULTS COUNT ---")
    print(f"Count: {result.get('results_count')}")
    
    # Print first result if any
    if result.get("results"):
        print(f"Example Result: {result['results'][0]['title']} (€{result['results'][0]['price']})")

if __name__ == "__main__":
    # Scenario 1: Straightforward search
    test_scenario("cerco scarpe nike sotto i 150 euro")
