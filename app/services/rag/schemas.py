from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import hashlib


def make_doc_id(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class RagDoc:
    doc_id: str
    text: str
    meta: Dict[str, Any]

    @staticmethod
    def from_text(text: str, meta: Optional[Dict[str, Any]] = None) -> "RagDoc":
        clean = " ".join((text or "").split()).strip()
        m = dict(meta or {})
        m["text"] = clean
        m["doc_id"] = m.get("doc_id") or make_doc_id(clean)
        return RagDoc(doc_id=m["doc_id"], text=clean, meta=m)