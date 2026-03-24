import os
import subprocess

def sanitize_environment():
    print("=== Sanitizzazione Ambiente E-commerce (Definitiva) ===")
    
    targets = ["python.exe", "uvicorn.exe", "node.exe"]
    
    # We use taskkill /F to be sure on Windows
    # IMPORTANT: This will kill the current process too if it's python.exe, 
    # but the shell will continue.
    for target in targets:
        try:
            print(f"Termino tutti i processi {target}...")
            # We use shell=True for taskkill on windows
            subprocess.run(f"taskkill /F /IM {target}", shell=True, capture_output=True)
        except Exception as e:
            print(f"Errore terminando {target}: {e}")
            
    print("\n[OK] Ambiente pulito. Ora assicurati di avere SOLO UNA finestra terminale attiva.")

if __name__ == "__main__":
    sanitize_environment()
