
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPEN_ROUTER_API_KEY")

async def test():
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PaoloPangallo/MCP_E-commerce",
        "X-Title": "LTR User Simulator",
    }
    payload = {
        "model": "google/gemma-3-12b-it:free",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.0
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Resp: {resp.text}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
