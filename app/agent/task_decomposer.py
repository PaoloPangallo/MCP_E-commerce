from __future__ import annotations

import json
import logging
from typing import List, Dict

from app.services.parser import call_gemini, call_ollama, extract_first_json_object

logger = logging.getLogger(__name__)


DECOMPOSER_PROMPT = """
You are an e-commerce task planner.

Decompose the user request into atomic tool tasks.

Available tools:

search_pipeline
input: {"query":"string"}

seller_pipeline
input: {"seller_name":"string","page":1,"limit":10}

Rules:
- Only produce tasks that correspond to the user request.
- If the request mentions a seller, create a seller_pipeline task.
- If the request mentions products, create a search_pipeline task.
- Multiple tasks are allowed.
- Do not invent seller names.

Return ONLY JSON:

{
 "tasks":[
   {"tool":"search_pipeline","input":{"query":"example"}},
   {"tool":"seller_pipeline","input":{"seller_name":"example"}}
 ]
}

No markdown.
"""


def decompose_query(query: str, llm_engine: str) -> List[Dict]:

    prompt = f"{DECOMPOSER_PROMPT}\n\nUser query:{query}"

    try:

        if llm_engine == "gemini":
            raw = call_gemini(prompt)

        elif llm_engine == "ollama":
            raw = call_ollama(prompt)

        else:
            return []

        json_text = extract_first_json_object(raw)

        if not json_text:
            return []

        data = json.loads(json_text)

        tasks = data.get("tasks")

        if not isinstance(tasks, list):
            return []

        return tasks

    except Exception as e:
        logger.warning("Task decomposition failed: %s", e)

    return []