
import asyncio
import logging
import os
import sys

# Set PYTHONPATH
sys.path.append(os.getcwd())

from app.agent.executor import ToolExecutor, ToolCall
from app.agent.tool_registry import ToolContext, get_tool_spec
from app.db.database import SessionLocal
from app.tools.seller_tool import execute_seller_tool
from app.services.seller_pipeline import run_seller_pipeline
from app.services.feedback import get_seller_feedback

logging.basicConfig(level=logging.INFO)

async def reproduce():
    db = SessionLocal()
    try:
        context = ToolContext(db=db, user=None, llm_engine="ollama")
        
        # Test 1: get_seller_feedback directly
        print("\n--- Test 1: get_seller_feedback ---")
        try:
            fb = await get_seller_feedback("giuseppe2181", limit=10)
            print(f"Type of feedbacks: {type(fb)}")
            if hasattr(fb, "get"):
                print("Wait, feedbacks is a dict? (Unexpected for a list)")
            else:
                print("Feedbacks is a list (Expected)")
        except Exception as e:
            print(f"Test 1 failed: {e}")

        # Test 2: run_seller_pipeline directly
        print("\n--- Test 2: run_seller_pipeline ---")
        try:
            payload = await run_seller_pipeline("giuseppe2181", limit=10)
            print(f"Type of payload: {type(payload)}")
            print(f"Payload keys: {payload.keys() if hasattr(payload, 'keys') else 'No keys'}")
        except Exception as e:
            print(f"Test 2 failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 3: execute_seller_tool directly
        print("\n--- Test 3: execute_seller_tool ---")
        try:
            res = await execute_seller_tool({"seller_name": "giuseppe2181"}, context)
            print(f"Type of res: {type(res)}")
        except Exception as e:
            print(f"Test 3 failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 4: Full ToolExecutor
        print("\n--- Test 4: ToolExecutor.execute ---")
        executor = ToolExecutor(context=context)
        tool_call = ToolCall(tool="analyze_seller", input={"seller_name": "giuseppe2181"})
        obs = await executor.execute(tool_call)
        print(f"Observation Status: {obs.status}")
        print(f"Observation Summary: {obs.summary}")
        if obs.error:
            print(f"Observation Error: {obs.error}")

    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(reproduce())
