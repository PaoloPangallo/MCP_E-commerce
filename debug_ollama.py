
import os
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(r"C:\Users\giova\MCP_E-commerce")
sys.path.append(str(root))

from app.services.parser import call_ollama

prompt = "Hello! Who are you?"
print(f"Calling Ollama with prompt: {prompt}")
res = call_ollama(prompt)
print(f"Result: {res}")
