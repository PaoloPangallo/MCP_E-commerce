import logging
from typing import Any, Dict

from app.services.ebay_metadata import get_marketplace_metadata

logger = logging.getLogger(__name__)


async def execute_metadata_tool(
    kwargs: Dict[str, Any],
    context: Any,
) -> Dict[str, Any]:
    """
    Retrieve eBay marketplace metadata (category policies).

    Args:
        kwargs: may contain 'policy_type', 'marketplace_id', 'category_id'
    """
    logger.info("EXECUTE metadata tool START")

    policy_type = kwargs.get("policy_type", "item_conditions")
    marketplace_id = kwargs.get("marketplace_id", "")
    category_id = kwargs.get("category_id") or None

    try:
        result = await get_marketplace_metadata(
            policy_type=policy_type,
            marketplace_id=marketplace_id,
            category_id=category_id,
        )

        has_error = "error" in result and not any(
            k for k in result if k not in ("error", "detail", "policy_type", "marketplace_id", "category_id")
        )

        return {
            "status": "error" if has_error else "ok",
            "policy_type": policy_type,
            "marketplace_id": result.get("marketplace_id", ""),
            "category_id": category_id,
            "results": result,
            "results_count": len(result) if not has_error else 0,
        }
    except Exception as exc:
        logger.exception("metadata tool FAILED")
        return {
            "status": "error",
            "policy_type": policy_type,
            "error": str(exc),
        }
