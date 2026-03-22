"""
app/llm/vision.py
-----------------
Image-to-text via Ollama Cloud Vision (Qwen-VL).
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from app.llm.client import call_ollama_cloud, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "Descrivi questo oggetto in modo estremamente dettagliato per una ricerca e-commerce. "
    "Indica marca, modello, colore, materiali e ogni dettaglio identificativo visibile. "
    "Rispondi solo con la descrizione."
)


async def describe_image_with_vision(image_b64: str) -> Optional[str]:
    """Interroga il modello Vision per ottenere una descrizione testuale dell'immagine."""
    try:
        # Rimuovi header base64 se presente (es. data:image/png;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        description = await call_ollama_cloud(
            VISION_PROMPT,
            model=OLLAMA_VISION_MODEL,
            images=[image_bytes],
        )

        if description:
            logger.info("VISION: described as: %s", description[:100] + "...")
            return description.strip()

    except Exception as exc:
        logger.error("VISION error: %s", exc)

    return None
