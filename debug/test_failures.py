import asyncio
import logging
import os
import sys

# Aggiungi il path del progetto per gli import
sys.path.insert(0, r"c:\Users\paolo\MCP_ECOM")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.models.listing import Listing
from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.services.parser import parse_query_service
from app.services.search_pipeline import run_search_pipeline
from app.services.ebay import search_items

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')

engine = create_engine("sqlite:///test_debug.db", connect_args={"check_same_thread": False})
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_query_filtering():
    """Test why 'iPhone 15 Pro usato' fails."""
    query = "Trova un mac e dimmi con cap 89030 quando mi arriva"
    parsed = parse_query_service(query, use_llm=True, include_meta=True)
    print("----- PARSED MAC QUERY -----")
    print(parsed)
    
    # Check directly ebay API response
    res = search_items(parsed_query=parsed, limit=5)
    print("eBay hits for Mac:", len(res))


async def test_agent_routing():
    """Test why LLM isn't calling shipping_costs/item_details."""
    query = "Trova un mac e dimmi con cap 89030 quando mi arriva"
    
    db = SessionLocal()
    agent = EbayReactAgent(db=db, strict_mcp=False, prefer_mcp=False)
    print("\n----- AGENT RUN FOR MACBOOK -----")
    req = AgentRequest(query=query, llm_engine="ollama", max_steps=5, return_trace=True)
    events = []
    async for event in agent.run_stream(req):
        if event.get("type") == "tool_start":
            print(f">> TOOL CALLED: {event.get('tool')} with {event.get('input')}")
        elif event.get("type") == "tool_result":
            print(f"<< TOOL RESULT: {event.get('tool')} status={event.get('status')}")
        elif event.get("type") == "final":
            print("<< FINAL EVENT")
        events.append(event)

    import json
    with open("test_trace.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

if __name__ == "__main__":
    test_query_filtering()
    asyncio.run(test_agent_routing())
