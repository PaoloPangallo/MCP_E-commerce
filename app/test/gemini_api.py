from google import genai
import os

# Forza il client a non usare parametri ambigui
client = genai.Client(
    api_key="AIzaSyC5BpduGhgHt0ceksGxnbffw-F3IVoXgn0",
)

try:
    # Usiamo il nome esatto che è apparso nella tua lista
    model_id = "gemini-2.5-flash"

    print(f"--- Test in corso con {model_id} ---")

    response = client.models.generate_content(
        model=model_id,
        contents="Ciao! Sono un assistente per il tuo e-commerce. Come posso aiutarti oggi?"
    )

    print("Risposta da Gemini:")
    print(response.text)

except Exception as e:
    print(f"Errore: {e}")