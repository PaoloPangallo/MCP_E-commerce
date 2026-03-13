from typing import Any

def clean_text(value: Any) -> str:
    """Consolidated utility to clean and strip text values."""
    return str(value or "").strip()

def compact_json_result(data: Any) -> Any:
    """Recursively clean empty strings/None from small JSON payloads if needed."""
    if isinstance(data, dict):
        return {k: compact_json_result(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [compact_json_result(i) for i in data if i is not None]
    return data
