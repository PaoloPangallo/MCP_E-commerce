import asyncio
import os
import sys
import json

# Add root folder to path so app.* works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.parser import parse_query_service

async def test():
    query = "Vorrei un iPhone rigenerato blu che stia sotto i 300 euro. Ma cercalo venduto da pegaso_italia per favore."
    res = await parse_query_service(query)
    print("Parsed JSON:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    asyncio.run(test())
