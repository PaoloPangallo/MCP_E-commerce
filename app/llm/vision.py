"""
app/llm/vision.py
-----------------
Image-to-text via Ollama Cloud Vision (Qwen-VL).
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, Optional

from app.llm.client import call_ollama_cloud, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

VISION_PROMPT = """Analizza questa immagine come un esperto e-commerce per una ricerca prodotto. Estrai i dettagli e restituisci ESCLUSIVAMENTE un resoconto in formato JSON.
Il JSON deve avere questa esatta struttura e chiavi in inglese:
{
  "description": "Descrizione estensiva in italiano (marca, modello, colore, materiale, dettagli identificativi)",
  "tags": ["keyword1", "keyword2", "keyword3", "tag4"],
  "brand": "Nome Brand se chiaramente visibile (oppure null)",
  "condition_clues": "Condizioni apparenti in italiano (es. nuovo con etichetta, apparentemente usurato, graffi visibili, etc) oppure null",
  "confidence": 0.95
}
NON inserire spiegazioni, introduzioni, markdown o backtick testuali. Produci SOLO output JSON valido.
"""

def _extract_json_from_text(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)


async def describe_image_with_vision(image_b64: str) -> Optional[Dict[str, Any]]:
    """Interroga il modello Vision per ottenere una descrizione JSON dell'immagine."""
    try:
        # Rimuovi header base64 se presente (es. data:image/png;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        raw_response = await call_ollama_cloud(
            VISION_PROMPT,
            model=OLLAMA_VISION_MODEL,
            images=[image_bytes],
        )

        if raw_response:
            try:
                parsed = _extract_json_from_text(raw_response)
                logger.info("VISION: Successfully parsed JSON response (brand: %s, tags: %s)", parsed.get("brand"), len(parsed.get("tags", [])))
                return parsed
            except Exception as e:
                logger.error("VISION JSON parsing failed: %s | Raw response: %s", e, raw_response)
                # Fallback to simple string wrapped in dict
                return {
                    "description": raw_response.strip(),
                    "tags": ["fallaback-vision"],
                    "brand": None,
                    "condition_clues": None,
                    "confidence": 0.5
                }

    except Exception as exc:
        logger.error("VISION error: %s", exc)

    return None
