import logging
import asyncio
from app.agent.memory import AgentMemory
from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.services.parser import load_dotenv
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.agent.ebay_agent")
logger.setLevel(logging.DEBUG)

async def reproduce():
    # Load env
    ROOT = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=ROOT / ".env")
    
    # Mock DB (since _finalize doesn't use it directly for conversation)
    db = None 
    
    agent = EbayReactAgent(db=db)
    
    # Create memory via MemoryService
    memory = agent.memory_service.hydrate_request_state(
        user_query="hey",
        user=None
    )
    memory.detected_intent = "conversation"
    
    print("Testing _finalize for 'hey'...")
    try:
        final_answer = await agent._finalize(memory, llm_engine="ollama")
        print(f"\nFINAL ANSWER: {final_answer}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce())
