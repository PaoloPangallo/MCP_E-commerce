import os
from dotenv import load_dotenv
from pathlib import Path

root = Path(__file__).resolve().parent
load_dotenv(dotenv_path=root / ".env")

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL")

print(f"OLLAMA_API_KEY present: {bool(OLLAMA_API_KEY)}")
print(f"OLLAMA_CLOUD_HOST: {OLLAMA_CLOUD_HOST}")
print(f"QDRANT_URL: {QDRANT_URL}")

if OLLAMA_API_KEY:
    OLLAMA_URL = f"{OLLAMA_CLOUD_HOST}/api/chat"
else:
    OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"

print(f"Computed OLLAMA_URL: {OLLAMA_URL}")
