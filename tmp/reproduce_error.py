
import asyncio
import logging
import os

# Set PYTHONPATH to the current directory
import sys
sys.path.append(os.getcwd())

from app.agent.executor import ToolExecutor, ToolCall
from app.db.database import SessionLocal
from app.agent.tool_registry import ToolContext

logging.basicConfig(level=logging.INFO)

async def test_search():
    db = SessionLocal()
    try:
        context = ToolContext(db=db, user=None, llm_engine="ollama")
        executor = ToolExecutor(context=context)
        
        tool_call = ToolCall(
            tool="search_products",
            input={"query": "nike"}
        )
        
        print("Testing search_products tool...")
        observation = await executor.execute(tool_call)
        print(f"Status: {observation.status}")
        print(f"Summary: {observation.summary}")
        if observation.error:
            print(f"Error: {observation.error}")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_search())
