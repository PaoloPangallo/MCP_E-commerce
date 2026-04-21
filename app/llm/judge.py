"""
app/llm/judge.py
----------------
LLM-as-a-Judge utilities for filtering and evaluating item lists.

Currently used to filter `similar_items` returned by eBay so that
only products relevant to the user's original search query are shown
in the detail panel.

Design principles:
- Fast: compact prompt, JSON-only response, strict max_tokens
- Safe: full no-op fallback if the LLM call fails or times out
- Pure: no side effects, just takes a list in and returns a filtered list
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.llm.client import call_ollama_cloud

logger = logging.getLogger(__name__)

# Maximum number of similar items to present to the judge.
# eBay may return more, but we cap the judge input to keep the prompt small.
_MAX_ITEMS_FOR_JUDGE = 8

_SYSTEM_PROMPT = (
    "Sei un assistente e-commerce esperto e molto selettivo. "
    "Il tuo compito è filtrare i prodotti simili mantenendo SOLO quelli che rispettano la MARCA e la CATEGORIA della richiesta dell'utente. "
    "Se l'utente cerca una marca specifica (es. Acer), scarta immediatamente prodotti di altre marche (es. Apple, Samsung). "
    "Rispondi SOLO con un array JSON di indici (interi, 0-indexed) dei prodotti pertinenti. "
    "Esempio: [0, 2]"
)


async def filter_similar_items_with_llm(
    items: List[Dict[str, Any]],
    user_query: str,
    main_item_title: str = "",
) -> List[Dict[str, Any]]:
    """
    Filters `items` (similar_items from eBay) keeping only those relevant
    to `user_query`. Returns the original list unchanged on any failure.

    Args:
        items:            List of normalized SearchItem dicts.
        user_query:       The user's original search query (raw text).
        main_item_title:  Title of the main item whose similars we are filtering.
                          Used as context so the judge doesn't filter out close matches.

    Returns:
        Filtered list. Falls back to the original list on error.
    """
    if not items or not user_query:
        return items

    # Cap the number of items to avoid ballooning the prompt
    candidates = items[:_MAX_ITEMS_FOR_JUDGE]

    # Build a compact summary for each candidate (title + price only)
    candidates_text = "\n".join(
        f"[{i}] {c.get('title', 'N/A')} — {c.get('price', '?')} {c.get('currency', 'EUR')}"
        for i, c in enumerate(candidates)
    )

    context_line = (
        f"Prodotto di riferimento: \"{main_item_title}\"\n"
        if main_item_title
        else ""
    )

    prompt = (
        f"Richiesta utente: \"{user_query}\"\n"
        f"{context_line}"
        f"Prodotti simili proposti:\n{candidates_text}\n\n"
        "Quali indici (0-based) sono DAVVERO pertinenti? "
        "REQUISITO RIGIDO: Mantieni solo prodotti della STESSA MARCA e CATEGORIA dell'oggetto di riferimento o della richiesta. "
        "Se l'oggetto di riferimento è Acer e vedi un iPhone, scartalo anche se ha la parola 'batteria'. "
        "Rispondi SOLO con l'array JSON degli indici."
    )

    try:
        raw = await call_ollama_cloud(
            prompt,
            system_prompt=_SYSTEM_PROMPT,
            stream=False,
        )

        if not raw:
            logger.warning("LLM judge returned empty response — skipping filter")
            return items

        # Strip markdown code fences if the model adds them
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        # Parse JSON array of indices
        indices: List[int] = json.loads(cleaned)

        if not isinstance(indices, list):
            raise ValueError(f"Expected list, got {type(indices)}")

        # Validate indices and keep only valid ones
        valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]

        if not valid_indices:
            # Judge said nothing is relevant — return empty (or fallback, your call)
            logger.info("LLM judge filtered out all %d similar items", len(candidates))
            return []

        filtered = [candidates[i] for i in valid_indices]
        # Append any items beyond _MAX_ITEMS_FOR_JUDGE that weren't judged (pass-through)
        filtered += items[_MAX_ITEMS_FOR_JUDGE:]

        logger.info(
            "LLM judge: kept %d/%d similar items for query=%r",
            len(filtered), len(items), user_query[:60],
        )
        return filtered

    except json.JSONDecodeError as exc:
        logger.warning("LLM judge: JSON parse error (%s) — raw=%r", exc, raw[:200] if raw else "")
        return items
    except Exception as exc:
        logger.warning("LLM judge: unexpected error (%s) — returning original list", exc)
        return items
