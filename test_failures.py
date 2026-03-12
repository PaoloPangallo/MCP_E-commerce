import asyncio
import os
import sys

# Aggiungi il path del progetto per gli import
sys.path.insert(0, r"c:\Users\paolo\MCP_ECOM")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.services.parser import parse_query_service
from app.services.search_pipeline import run_search_pipeline
from app.services.ebay import search_items

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_query_filtering():
    """Test why 'iPhone 15 Pro usato' fails."""
    query = "Trova un iPhone 15 Pro usato e dimmi quanto costa spedirlo a Roma (CAP 00100)."
    parsed = parse_query_service(query, use_llm=True, include_meta=True)
    print("----- PARSED IPHONE QUERY -----")
    print(parsed)
    
    # Check directly ebay API response
    res = search_items(parsed_query=parsed, limit=5)
    print("eBay hits for iPhone:", len(res))


async def test_agent_routing():
    """Test why LLM isn't calling item_details."""
    query = "Cerca un portatile MacBook Pro M3 e dimmi esattamente quali sono le specifiche tecniche del primo risultato, specialmente la RAM e il materiale della scocca."
    
    db = SessionLocal()
    agent = EbayReactAgent(db=db, strict_mcp=False, prefer_mcp=False)
    req = AgentRequest(query=query, llm_engine="ollama", max_steps=5, return_trace=True)
    
    print("\n----- AGENT RUN FOR MACBOOK -----")
    async for event in agent.run_stream(req):
        if event.get("type") == "tool_start":
            print(f">> TOOL CALLED: {event.get('tool')} with {event.get('input')}")
        elif event.get("type") == "tool_result":
            print(f"<< TOOL RESULT: {event.get('tool')} status={event.get('status')}")
        elif event.get("type") == "final":
            print("<< FINAL EVENT")

if __name__ == "__main__":
    test_query_filtering()
    asyncio.run(test_agent_routing())

