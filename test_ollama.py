import os
from dotenv import load_dotenv

# Forza il caricamento del token env (incluso EBAY_USER_TOKEN)
load_dotenv()

from app.services.agent_ollama import ask_ollama_agent

prompt = "Sto cercando una scrivania per pc. Costerà massimo 100 euro. Mostrami qualcosa di interessante e dimmi se il venditore da cui compro è affidabile."

print("=== INIZIO TEST AGENTE ===")
print("Query:", prompt)
print("-" * 50)
risposta = ask_ollama_agent(prompt)
print("-" * 50)
print("Risposta finale dell'Agente:")
print(risposta)
print("=== FINE TEST ===")
