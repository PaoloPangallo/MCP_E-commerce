
import asyncio
import os
import sys
import inspect
import functools

# PYTHONPATH
sys.path.append(os.getcwd())

from app.agent.tool_registry import TOOLS

async def main():
    print(f"{'TOOL NAME':<20} | {'IS_CORO':<7} | {'TYPE':<20} | {'WRAPPED':<7}")
    print("-" * 60)
    for name, spec in TOOLS.items():
        is_coro = asyncio.iscoroutinefunction(spec.executor)
        exec_type = str(type(spec.executor))
        has_wrapped = hasattr(spec.executor, "__wrapped__")
        
        print(f"{name:<20} | {str(is_coro):<7} | {exec_type:<20} | {str(has_wrapped):<7}")
        
        if name == "analyze_seller":
            print(f"  -> Module: {spec.executor.__module__}")
            if has_wrapped:
                print(f"  -> Wrapped: {spec.executor.__wrapped__}")

if __name__ == "__main__":
    asyncio.run(main())
