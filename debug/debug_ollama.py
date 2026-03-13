import os
import requests
from dotenv import load_dotenv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env")

model = os.getenv("OLLAMA_MODEL", "not_found")
print(f"DEBUG: OLLAMA_MODEL='{model}'")

try:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": "test", "stream": False},
        timeout=5
    )
    print(f"DEBUG: HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"DEBUG: Response: {resp.text}")
except Exception as e:
    print(f"DEBUG: Error: {e}")
