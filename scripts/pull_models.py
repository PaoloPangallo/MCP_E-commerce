import subprocess
import sys

MODELS = [
    "llama3:8b",          # Spec Nerd
    "mistral:7b",         # Vintage Hunter
    "gemma2:9b",         # Designer
    "phi3:mini",         # Budget Renovator
    "qwen2:7b",           # PC Master Race
    "llama3.1:8b",       # Early Adopter
    "mistral:7b-instruct",# Practical DIYer
    "gemma:2b",          # Minimalist
    "phi3:medium",       # Value Hunter
    "qwen2:1.5b",         # Cheapskate
    "tinyllama",         # Hobbyist Gamer
    "stable-zephyr:3b",  # Aesthetic Homemaker
    "orca-mini:3b",      # Refurbished Expert
    "openhermes",        # Luxury Buyer
    "neural-chat"        # Mint Collector
]

def pull_models():
    print(f"--- Starting Pull for {len(MODELS)} models ---")
    for model in MODELS:
        print(f"Pulling {model}...")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"Successfully pulled {model}")
        except subprocess.CalledProcessError as e:
            print(f"Error pulling {model}: {e}")
        except FileNotFoundError:
            print("Error: 'ollama' command not found. Is Ollama installed?")
            sys.exit(1)
    print("--- Finished ---")

if __name__ == "__main__":
    pull_models()
