"""
app/llm/client.py
-----------------
Single source of truth per le chiamate LLM.
Poiché il progetto usa solo Ollama Cloud, questo modulo espone
esclusivamente call_ollama_cloud e le funzioni wrapper pubbliche.

Tutti i moduli (parser, agent, planner) importano da qui,
NON da services/parser.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config (letti da env, nessun default hardcoded per i segreti) ────────────

OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_CLOUD_HOST: str = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com").rstrip("/")
OLLAMA_CLOUD_MODEL: str = os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")
OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:235b-cloud")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "45"))


def _get_ollama_client():
    """Ritorna un AsyncClient Ollama configurato per Ollama Cloud."""
    try:
        from ollama import AsyncClient  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "Libreria 'ollama' non installata. Esegui: pip install ollama"
        )

    if not OLLAMA_API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY non impostata — Ollama Cloud non disponibile."
        )

    return AsyncClient(
        host=OLLAMA_CLOUD_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )


# ── Core call ────────────────────────────────────────────────────────────────

async def call_ollama_cloud(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    images: Optional[List[bytes]] = None,
    model: Optional[str] = None,
) -> Any:
    """
    Chiama Ollama Cloud via SDK ufficiale.

    - stream=False → restituisce Optional[str]
    - stream=True  → restituisce AsyncGenerator[str, None]
    - images       → attiva modalità vision (multimodale)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_message: dict = {"role": "user", "content": prompt}
    if images:
        user_message["images"] = images
    messages.append(user_message)

    selected_model = model or OLLAMA_CLOUD_MODEL

    try:
        client = _get_ollama_client()
    except RuntimeError as exc:
        logger.error("LLM client init failed: %s", exc)
        return None

    if not stream:
        try:
            # Add explicit timeout for non-streaming calls
            response = await asyncio.wait_for(
                client.chat(
                    model=selected_model,
                    messages=messages,
                    options={"num_predict": 4096},
                ),
                timeout=OLLAMA_TIMEOUT
            )
            content = (response.message.content or "").strip()
            return content if content else None
        except asyncio.TimeoutError:
            logger.error("Ollama Cloud call timed out after %ss", OLLAMA_TIMEOUT)
            return None
        except Exception as exc:
            logger.warning("Ollama Cloud error: %s", exc)
            return None

    # ── Streaming ────────────────────────────────────────────────────────────
    async def _generator() -> AsyncGenerator[str, None]:
        first_chunk = True
        try:
            # The client.chat(stream=True) returns an async generator. 
            # We apply timeout to getting the stream itself.
            response_stream = await asyncio.wait_for(
                client.chat(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    options={"num_predict": 4096},
                ),
                timeout=OLLAMA_TIMEOUT
            )
            
            async for part in response_stream:
                content = part.message.content
                if content:
                    if first_chunk:
                        logger.info("Ollama Cloud: First chunk received")
                        first_chunk = False
                    yield content
        except asyncio.TimeoutError:
            logger.error("Ollama Cloud stream initiation timed out after %ss", OLLAMA_TIMEOUT)
            yield ""
        except Exception as exc:
            logger.warning("Ollama Cloud streaming error: %s", exc)
            yield ""

    return _generator()


# ── Public API ───────────────────────────────────────────────────────────────

async def call_llm(*args, **kwargs) -> Tuple[Optional[str], str]:
    """
    Entry point unificato per chiamate LLM non-streaming.
    Cattura tutti gli argomenti per debuggare l'errore '2 positional given'.
    """
    # Debug: logger.info("DEBUG: call_llm args=%s kwargs=%s", args, kwargs)
    
    # Estraiamo prompt e parametri dai vari formati possibili
    prompt = kwargs.get("prompt")
    system_prompt = kwargs.get("system_prompt")
    
    if args:
        if not prompt: 
            prompt = args[0]
        if len(args) > 1 and not kwargs.get("llm_engine"):
            # Il secondo argomento è l'engine (ignorato)
            pass
            
    if not prompt:
        logger.error("call_llm: prompt non trovato tra args=%s o kwargs=%s", args, kwargs)
        return None, "error"

    result = await call_ollama_cloud(prompt, system_prompt=system_prompt)
    if result:
        return result, "ollama_cloud"
    return None, "ollama_cloud"


async def call_llm_stream(prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Entry point unificato per chiamate LLM in streaming.
    """
    gen = await call_ollama_cloud(prompt, system_prompt=system_prompt, stream=True)
    if gen is not None:
        async for chunk in gen:
            yield chunk
