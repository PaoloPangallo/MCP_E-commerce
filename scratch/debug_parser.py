import asyncio
import json
import os
from app.services.parser import parse_query_service

async def debug_parser():
    query = "ciao, possiamo cercare una batteria per acer aspire 5?"
    # Simulating a context heavily polluted by iPhones
    context = "iPhone 13 | Apple phone | smartphone"
    
    print(f"DEBUGGING PARSER")
    print(f"Query: {query}")
    print(f"Context: {context}")
    print("-" * 30)
    
    result = await parse_query_service(query, use_llm=True, context_info=context)
    
    print(f"PASRED RESULT:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(debug_parser())
