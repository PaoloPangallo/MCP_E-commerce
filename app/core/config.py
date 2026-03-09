# app/core/config.py
import os

OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "mistral-nemo:12b")
PARSER_MODEL = os.getenv("PARSER_MODEL", "qwen2.5-coder:7b")
TOOL_CALLER_MODEL = os.getenv("TOOL_CALLER_MODEL", "qwen2.5-coder:7b")

# Altre configurazioni globali
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "1"))
