import asyncio
import logging
from app.mcp.client import MCPToolClient
from app.agent.planner import ReactPlanner, AgentMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify")

async def verify():
    print("--- Verificando MCP Client (Native Mode) ---")
    client = MCPToolClient(enabled=True)
    try:
        catalog = await client.get_tool_schemas_async()
        print(f"Success! Discovered {len(catalog)} tools natively.")
        for name in catalog:
            print(f" - {name}")
    except Exception as e:
        print(f"FAILED Native MCP verification: {e}")

    print("\n--- Verificando Planner (Sans Fallback) ---")
    # Verifichiamo che non ci siano riferimenti a funzioni mancanti
    try:
        from app.agent.planner import get_tool_catalog
        print("Wait, get_tool_catalog STILL EXISTS?! FAILED.")
    except ImportError:
        print("Success! get_tool_catalog is gone as requested.")
    except NameError:
        print("Success! get_tool_catalog is gone as requested.")
    except Exception as e:
        print(f"Info: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
