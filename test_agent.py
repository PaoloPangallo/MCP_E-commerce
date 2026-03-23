import asyncio
from app.db.database import SessionLocal
from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest

async def test():
    db = SessionLocal()
    agent = EbayReactAgent(db=db, prefer_mcp=True, strict_mcp=True)
    print("Agent initialized.")
    
    req = AgentRequest(query="mi cerchi iphone 13 usato?", return_trace=True, llm_engine="ollama")
    
    result = await agent.run(req)
    print("--- RESULT ---")
    print(result.final_answer)
    
    if result.agent_trace:
        print("--- TRACE ---")
        for step in result.agent_trace:
            if getattr(step, 'tool', None):
                print(f"Tool: {step.tool} | Input: {step.input}")
            
    db.close()

if __name__ == "__main__":
    asyncio.run(test())
