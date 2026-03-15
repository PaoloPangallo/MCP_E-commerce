import asyncio
import json
from app.services.parser import parse_query_service

async def test_parser():
    queries = [
        "iphone 13 blu",
        "macbook con 16gb ram",
        "scarpe nike taglia 42 rossi"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        result = await parse_query_service(q)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_parser())
