import os
from dotenv import load_dotenv

load_dotenv()

vision_model = os.getenv("OLLAMA_VISION_MODEL")
print(f"OLLAMA_VISION_MODEL: {vision_model}")

if vision_model == "qwen3-vl:235b-cloud":
    print("SUCCESS: Vision model is correctly configured.")
else:
    print("FAILURE: Vision model is NOT correctly configured.")
