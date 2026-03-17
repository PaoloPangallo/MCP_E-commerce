"""
eBay Sell Metadata API v1 — marketplace policy retrieval.

Uses the same OAuth token (client_credentials) from ebay.py.
Base URL: https://api.ebay.com/sell/metadata/v1/marketplace/{marketplace_id}/...
"""
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EBAY_ENV = os.getenv("EBAY_ENV", "sandbox").strip().lower()
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_IT").strip()

if EBAY_ENV == "production":
    METADATA_BASE_URL = "https://api.ebay.com/sell/metadata/v1/marketplace"
else:
    METADATA_BASE_URL = "https://api.sandbox.ebay.com/sell/metadata/v1/marketplace"

# Supported policy types mapped to endpoint suffixes
_POLICY_ENDPOINTS = {
    "item_conditions": "get_item_condition_policies",
    "return_policies": "get_return_policies",
    "listing_structure": "get_listing_structure_policies",
}

VALID_POLICY_TYPES = list(_POLICY_ENDPOINTS.keys())


async def _fetch_metadata(
    endpoint_suffix: str,
    marketplace_id: str,
    category_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generic fetcher for any Metadata API endpoint."""
    from app.services.ebay import _get_oauth_token, get_client, REQUEST_TIMEOUT

    token = await _get_oauth_token()
    client = get_client()

    url = f"{METADATA_BASE_URL}/{marketplace_id}/{endpoint_suffix}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    params: Dict[str, str] = {}
    if category_id:
        params["filter"] = f"categoryId:{{{category_id}}}"

    try:
        response = await client.get(
            url, headers=headers, params=params, timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 401:
            token = await _get_oauth_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.get(
                url, headers=headers, params=params, timeout=REQUEST_TIMEOUT
            )

        if response.status_code != 200:
            logger.warning(
                "METADATA API ERROR | endpoint=%s | status=%s | body=%s",
                endpoint_suffix,
                response.status_code,
                response.text[:500],
            )
            return {
                "error": f"eBay Metadata API returned {response.status_code}",
                "detail": response.text[:300],
            }

        return response.json()

    except Exception as exc:
        logger.exception("METADATA API EXCEPTION | endpoint=%s", endpoint_suffix)
        return {"error": str(exc)}


# ============================================================
# PUBLIC API
# ============================================================


async def get_item_condition_policies(
    marketplace_id: str = "",
    category_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve supported item conditions for a marketplace/category."""
    mkt = marketplace_id or EBAY_MARKETPLACE_ID
    logger.info("METADATA get_item_condition_policies | mkt=%s cat=%s", mkt, category_id)
    return await _fetch_metadata("get_item_condition_policies", mkt, category_id)


async def get_return_policies(
    marketplace_id: str = "",
    category_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve return policy requirements for a marketplace/category."""
    mkt = marketplace_id or EBAY_MARKETPLACE_ID
    logger.info("METADATA get_return_policies | mkt=%s cat=%s", mkt, category_id)
    return await _fetch_metadata("get_return_policies", mkt, category_id)


async def get_listing_structure_policies(
    marketplace_id: str = "",
    category_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve listing structure (multi-variation support) for a marketplace/category."""
    mkt = marketplace_id or EBAY_MARKETPLACE_ID
    logger.info("METADATA get_listing_structure_policies | mkt=%s cat=%s", mkt, category_id)
    return await _fetch_metadata("get_listing_structure_policies", mkt, category_id)


async def get_marketplace_metadata(
    policy_type: str = "item_conditions",
    marketplace_id: str = "",
    category_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified dispatcher for Metadata API calls.

    Args:
        policy_type: One of 'item_conditions', 'return_policies', 'listing_structure'
        marketplace_id: eBay marketplace ID (default from env)
        category_id: Optional category ID to filter results
    """
    policy_type = (policy_type or "item_conditions").strip().lower()
    endpoint = _POLICY_ENDPOINTS.get(policy_type)

    if not endpoint:
        return {
            "error": f"Invalid policy_type '{policy_type}'. Valid: {VALID_POLICY_TYPES}",
        }

    mkt = marketplace_id or EBAY_MARKETPLACE_ID
    logger.info(
        "METADATA get_marketplace_metadata | type=%s mkt=%s cat=%s",
        policy_type, mkt, category_id,
    )

    result = await _fetch_metadata(endpoint, mkt, category_id)
    result["policy_type"] = policy_type
    result["marketplace_id"] = mkt
    if category_id:
        result["category_id"] = category_id

    return result
