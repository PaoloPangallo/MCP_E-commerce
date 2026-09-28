import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Copy .env.example to .env and set a valid key before running this test."
    )

client = genai.Client(api_key=api_key)
model_id = "gemini-2.5-flash"

try:
    print(f"--- Testing {model_id} ---")
    response = client.models.generate_content(
        model=model_id,
        contents="Ciao! Sono un assistente per il tuo e-commerce. Come posso aiutarti oggi?",
    )
    print("Gemini response:")
    print(response.text)
except Exception as exc:
    print(f"Gemini test failed: {exc}")
    raise
