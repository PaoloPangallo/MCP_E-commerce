import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

def test_chat_ollama():
    ROOT = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=ROOT / ".env")
    
    model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "250"))
    
    messages = [
        {"role": "system", "content": "Sei ebayGPT, un assistente e-commerce amichevole."},
        {"role": "user", "content": "ciao"}
    ]
    
    print(f"Calling Ollama Chat ({model})...")
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": 256}
            },
            timeout=timeout
        )
        print(f"HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            message = data.get("message", {})
            print("\nMESSAGE KEYS:", list(message.keys()))
            if "thought" in message:
                print("\nTHOUGHT CONTENT (truncated):")
                print(message["thought"][:500])
            print("\nMESSAGE CONTENT:")
            print(message.get("content"))
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_chat_ollama()
