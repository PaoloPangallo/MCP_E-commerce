
import asyncio
import logging
import os
import sys
import inspect

# Set PYTHONPATH
sys.path.append(os.getcwd())

from app.agent.executor import ToolExecutor, ToolCall
from app.agent.tool_registry import TOOLS, ToolContext, get_tool_spec
from app.db.database import SessionLocal

logging.basicConfig(level=logging.INFO)

async def check_registry():
    spec = TOOLS.get("analyze_seller")
    if not spec:
        print("Error: analyze_seller not found in registry")
        return

    print(f"Tool: {spec.name}")
    print(f"Executor: {spec.executor}")
    is_coro = asyncio.iscoroutinefunction(spec.executor)
    print(f"Is coroutine function: {is_coro}")
    
    # Check if nested
    if hasattr(spec.executor, "__wrapped__"):
        print(f"Wrapped function: {spec.executor.__wrapped__}")
        print(f"Is wrapped coro: {asyncio.iscoroutinefunction(spec.executor.__wrapped__)}")

    db = SessionLocal()
    try:
        context = ToolContext(db=db, user=None, llm_engine="ollama")
        executor = ToolExecutor(context=context)
        
        # Simula _execute_once
        print("\n--- Manual execution simulation ---")
        if is_coro:
            res = await spec.executor({"seller_name": "giuseppe2181"}, context)
        else:
            res = spec.executor({"seller_name": "giuseppe2181"}, context)
        
        print(f"Result type: {type(res)}")
        if inspect.iscoroutine(res):
            print("WARNING: Execution returned a COROUTINE object!")
            # Resolve it just to see what happens
            actual_res = await res
            print(f"Resolved result type: {type(actual_res)}")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(check_registry())
