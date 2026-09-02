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

from app.llm.client import call_ollama_cloud, OLLAMA_TIMEOUT, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

VISION_PROMPT = """Analizza questa immagine come un esperto e-commerce per una ricerca prodotto. Estrai i dettagli e restituisci ESCLUSIVAMENTE un resoconto in formato JSON.
Il JSON deve avere questa esatta struttura e chiavi in inglese:
{
  "description": "Descrizione estensiva in italiano (marca, modello, colore, materiale, dettagli identificativi)",
  "tags": ["keyword1", "keyword2", "keyword3", "keyword4"],
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
        payload_chars = len(image_b64 or "")
        had_header = "," in (image_b64 or "")

        # Rimuovi header base64 se presente (es. data:image/png;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)
        logger.info(
            "[VISION-DIAG] model=%s payload_chars=%s had_header=%s decoded_bytes=%s magic=%s",
            OLLAMA_VISION_MODEL,
            payload_chars,
            had_header,
            len(image_bytes),
            image_bytes[:8].hex(),
        )

        raw_response = await call_ollama_cloud(
            VISION_PROMPT,
            model=OLLAMA_VISION_MODEL,
            images=[image_bytes],
        )

        if raw_response:
            logger.info(
                "[VISION-DIAG] raw_response chars=%s preview=%r",
                len(raw_response),
                raw_response[:500],
            )
            try:
                parsed = _extract_json_from_text(raw_response)
                logger.info("VISION: Successfully parsed JSON response (brand: %s, tags: %s)", parsed.get("brand"), len(parsed.get("tags", [])))
                return parsed
            except Exception as e:
                logger.error("VISION JSON parsing failed: %s | Raw response: %s", e, raw_response)
                logger.error("[VISION-DIAG] outcome=json_parse_failed (uso il fallback a stringa)")
                # Fallback to simple string wrapped in dict
                return {
                    "description": raw_response.strip(),
                    "tags": [],
                    "brand": None,
                    "condition_clues": None,
                    "confidence": 0.5,
                    "_parse_failed": True,
                }

        logger.error(
            "[VISION-DIAG] outcome=empty_response — call_ollama_cloud ha restituito %r. "
            "Cause tipiche: OLLAMA_API_KEY mancante, modello %s non disponibile "
            "sull'account Ollama Cloud, oppure timeout di %ss.",
            raw_response,
            OLLAMA_VISION_MODEL,
            OLLAMA_TIMEOUT,
        )

    except Exception as exc:
        logger.error("VISION error: %s", exc)
        logger.error(
            "[VISION-DIAG] outcome=exception type=%s", type(exc).__name__, exc_info=True
        )

    return None
