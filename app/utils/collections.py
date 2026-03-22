"""
app/utils/collections.py
------------------------
Utility functions per operazioni su collezioni.
Centralizziamo qui la deduplicazione (era duplicata in ebay.py e search_pipeline.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, TypeVar

T = TypeVar("T", bound=Dict[str, Any])


def dedupe_by_ebay_id(items: List[T]) -> List[T]:
    """
    Deduplicazione di una lista di prodotti eBay per `ebay_id`.
    Mantiene il primo occorrenza (ordine originale preservato).
    Ignora items senza ebay_id.
    """
    seen: Set[str] = set()
    result: List[T] = []

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            continue
        if ebay_id in seen:
            continue
        seen.add(ebay_id)
        result.append(item)

    return result


def dedupe_keep_order(items: List[Any]) -> List[Any]:
    """
    Deduplicazione generica (stringhe o dict) mantenendo l'ordine.
    Usata per brand, constraints, preferences nel parser.
    """
    import json
    seen = set()
    out = []

    for item in items:
        key = item.lower() if isinstance(item, str) else json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out
