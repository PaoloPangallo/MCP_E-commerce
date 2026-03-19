import asyncio
import os
import sys
from dotenv import load_dotenv

# Aggiunge il path della root del progetto
sys.path.append(os.getcwd())

load_dotenv()

async def test_vision():
    from app.services.parser import describe_image_with_vision
    
    print("🚀 Inizio test Vision (Qwen-VL su Ollama Cloud)...")
    
    # Mock di una piccolissima immagine PNG (1x1 pixel trasparente)
    mock_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    try:
        print("Interrogazione Ollama Cloud...")
        description = await describe_image_with_vision(mock_image_b64)
        
        if description:
            print(f"\n✅ TEST SUPERATO!")
            print(f"Descrizione IA: {description}")
        else:
            print("\n❌ ERRORE: Nessuna descrizione ricevuta (forse il modello non è pulllato?)")
            
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_vision())
