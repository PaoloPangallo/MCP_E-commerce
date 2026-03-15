import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

EBAY_USER_TOKEN = os.getenv("EBAY_USER_TOKEN")
TRADING_URL = os.getenv("EBAY_TRADING_URL", "https://api.ebay.com/ws/api.dll")
COMPATIBILITY_LEVEL = os.getenv("EBAY_COMPATIBILITY_LEVEL", "1451")
REQUEST_TIMEOUT = int(os.getenv("EBAY_FEEDBACK_TIMEOUT", "10"))
SITE_ID = os.getenv("EBAY_SITE_ID", "101")

_PER_PAGE = 50

_NS = {"e": "urn:ebay:apis:eBLBaseComponents"}


def _clean_username(username: str) -> str:
    return (username or "").strip()


def _build_headers() -> Dict[str, str]:
    return {
        "X-EBAY-API-CALL-NAME": "GetFeedback",
        "X-EBAY-API-COMPATIBILITY-LEVEL": COMPATIBILITY_LEVEL,
        "X-EBAY-API-SITEID": SITE_ID,
        "Content-Type": "text/xml",
    }


def _build_body(username: str, page: int, per_page: int) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<GetFeedbackRequest xmlns="urn:ebay:apis:eBLBaseComponents">
  <RequesterCredentials>
    <eBayAuthToken>{EBAY_USER_TOKEN}</eBayAuthToken>
  </RequesterCredentials>
  <UserID>{username}</UserID>
  <DetailLevel>ReturnAll</DetailLevel>
  <Pagination>
    <EntriesPerPage>{per_page}</EntriesPerPage>
    <PageNumber>{page}</PageNumber>
  </Pagination>
</GetFeedbackRequest>"""


def _safe_find_text(node: ET.Element, path: str, default: str = "") -> str:
    try:
        return node.findtext(path, default=default, namespaces=_NS) or default
    except Exception:
        return default


def _parse_ack_and_errors(root: ET.Element) -> Optional[str]:
    ack = _safe_find_text(root, ".//e:Ack", "")
    if ack and ack.lower() in {"failure", "partialfailure"}:
        short_msg = _safe_find_text(root, ".//e:Errors/e:ShortMessage", "")
        long_msg = _safe_find_text(root, ".//e:Errors/e:LongMessage", "")
        return long_msg or short_msg or f"eBay Trading API ack={ack}"
    return None


def _parse_feedback_page(xml_text: str) -> List[Dict]:
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        logger.warning("Feedback XML parse failed: %s", e)
        return []

    api_error = _parse_ack_and_errors(root)
    if api_error:
        logger.warning("eBay GetFeedback returned error: %s", api_error)
        return []

    feedbacks: List[Dict] = []

    for fb in root.findall(".//e:FeedbackDetail", _NS):
        raw_rating = _safe_find_text(fb, "e:CommentType", "Neutral")
        
        # Mapping eBay string to numeric rating for frontend/NLP consistency
        # 5 = Positive, 3 = Neutral, 1 = Negative
        rating_map = {
            "Positive": 5,
            "Neutral": 3,
            "Negative": 1
        }
        numeric_rating = rating_map.get(raw_rating, 3)

        feedbacks.append(
            {
                "user": _safe_find_text(fb, "e:CommentingUser", ""),
                "rating": numeric_rating,
                "comment": _safe_find_text(fb, "e:CommentText", ""),
                "time": _safe_find_text(fb, "e:CommentTime", ""),
            }
        )

    return feedbacks


async def fetch_feedback_page(
    client: httpx.AsyncClient,
    username: str,
    page: int,
    per_page: int = _PER_PAGE
) -> List[Dict]:
    """
    Fetch a single feedback page using an async client.
    """
    username = _clean_username(username)
    if not username or not EBAY_USER_TOKEN:
        return []

    headers = _build_headers()
    body = _build_body(username, page, per_page)

    for attempt in range(2):
        try:
            response = await client.post(
                TRADING_URL,
                headers=headers,
                content=body.encode("utf-8"),
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                logger.warning("GetFeedback HTTP %s for %s", response.status_code, username)
                return []

            return _parse_feedback_page(response.text)

        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.warning("GetFeedback error for %s p%s: %s", username, page, e)

    return []


async def get_seller_feedback(username: str, limit: int = 200) -> List[Dict]:
    """
    Fetch seller feedback in parallel pages.
    """
    username = _clean_username(username)
    if not username:
        return []

    limit = max(1, min(int(limit), 500))
    per_page = min(_PER_PAGE, limit)
    max_pages = (limit + per_page - 1) // per_page

    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_feedback_page(client, username, page, per_page)
            for page in range(1, max_pages + 1)
        ]
        results = await asyncio.gather(*tasks)

    all_feedback: List[Dict] = []
    for page_results in results:
        if page_results:
            all_feedback.extend(page_results)
            if len(all_feedback) >= limit:
                break

    return all_feedback[:limit]


def clear_feedback_cache() -> None:
    get_seller_feedback.cache_clear()