
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_CLOUD_API_KEY")

async def test():
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a ranker."},
            {"role": "user", "content": "Rank these: 1. ID=A, 2. ID=B"}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Resp: {resp.text}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
