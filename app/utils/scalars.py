import math
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

def sanitize_scalars(obj: Any) -> Any:
    """
    Recursively converts NumPy types, NaNs, and Infs into Python-native types 
    for JSON serialization and Pydantic compatibility.
    """
    if isinstance(obj, dict):
        return {k: sanitize_scalars(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_scalars(i) for i in obj]
    elif np is not None and isinstance(obj, np.ndarray):
        # If it's a single-element array, convert to scalar. Otherwise, list.
        if obj.size == 1:
            return sanitize_scalars(obj.item())
        return [sanitize_scalars(i) for i in obj.tolist()]
    elif np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    elif np is not None and isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')): # NumPy scalars or size-1 arrays
        try:
            return sanitize_scalars(obj.item())
        except (ValueError, TypeError):
            if hasattr(obj, 'tolist'):
                return sanitize_scalars(obj.tolist())
            return str(obj)
    return obj
