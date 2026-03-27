import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.planner import ReactPlanner
class DummyMemory:
    def __init__(self, query: str, tasks: list = None):
        self.user_query = query
        self.tasks = tasks or []
        self.tool_call_counts = {}
        self.observations = []
        self.last_seller_name = None
        self.detected_intent = None

    def has_pending_tasks(self):
        return len(self.tasks) > 0
    
    def peek_task(self):
        return self.tasks[0] if self.tasks else None

    def tool_call_count(self, tool):
        return 0

    def has_terminal_state(self, tool):
        return False

async def main():
    planner = ReactPlanner(llm_engine="rule_based")
    
    # Simula il caso dello screenshot: Query con Playwright ma con task queue (search_products) già pronta
    memory = DummyMemory(
        "Mostra iPhone 15 con Playwright", 
        tasks=[{"tool": "search_products", "input": {"query": "iphone 15"}}]
    )
    
    # Prima del fix questo avrebbe ritornato 'product_search' a causa della coda task.
    # Ora deve tornare 'playwright_search'.
    decision = await planner.decide(memory, 1, 4)
    intent = getattr(decision, "intent", None)
    tool = decision.action.tool if decision.action else "None"
    
    print(f"Query: {memory.user_query}")
    print(f"Tasks: {memory.tasks}")
    print(f"Final Intent: {intent}")
    print(f"Action Tool: {tool}")
    
    if intent == "playwright_search" and tool == "ebay_scrape":
        print("\n✅ SUCCESS: Overridden task queue with Playwright!")
    else:
        print("\n❌ FAILURE: Task queue still winning or routing failed.")

if __name__ == "__main__":
    asyncio.run(main())
