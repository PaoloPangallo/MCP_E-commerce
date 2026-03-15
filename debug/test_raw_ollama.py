import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

def test_raw_ollama():
    ROOT = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=ROOT / ".env")
    
    model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "250"))
    
    prompt = "Sei ebayGPT. Rispondi in italiano. Messaggio utente: hey"
    
    print(f"Calling Ollama ({model})...")
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100}
            },
            timeout=timeout
        )
        print(f"HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print("\nRAW RESPONSE:")
            print(json.dumps(data, indent=2))
            print("\nTEXT CONTENT:")
            print(data.get("response"))
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_raw_ollama()
