import asyncio
import logging
import os
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils
from typing import Dict, List, Optional, Any

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


def _build_headers(call_name: str) -> Dict[str, str]:
    return {
        "X-EBAY-API-CALL-NAME": call_name,
        "X-EBAY-API-COMPATIBILITY-LEVEL": COMPATIBILITY_LEVEL,
        "X-EBAY-API-SITEID": SITE_ID,
        "Content-Type": "text/xml",
    }


def _build_body(username: str, page: int, per_page: int) -> str:
    safe_username = saxutils.escape(username)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<GetFeedbackRequest xmlns="urn:ebay:apis:eBLBaseComponents">
  <RequesterCredentials>
    <eBayAuthToken>{EBAY_USER_TOKEN}</eBayAuthToken>
  </RequesterCredentials>
  <UserID>{safe_username}</UserID>
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
                "item_title": _safe_find_text(fb, "e:ItemTitle", ""),
                "item_id": _safe_find_text(fb, "e:ItemID", ""),
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

    headers = _build_headers("GetFeedback")
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


async def get_user_details(username: str) -> Dict[str, Any]:
    """
    Fetch basic user details using GetUser.
    """
    username = _clean_username(username)
    if not username or not EBAY_USER_TOKEN:
        return {}

    headers = _build_headers("GetUser")
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<GetUserRequest xmlns="urn:ebay:apis:eBLBaseComponents">
  <RequesterCredentials>
    <eBayAuthToken>{EBAY_USER_TOKEN}</eBayAuthToken>
  </RequesterCredentials>
  <UserID>{username}</UserID>
  <DetailLevel>ReturnAll</DetailLevel>
</GetUserRequest>"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(TRADING_URL, headers=headers, content=body.encode("utf-8"), timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                return {}
            
            root = ET.fromstring(response.text)
            user_node = root.find(".//e:User", _NS)
            if user_node is None:
                return {}

            reg_date = _safe_find_text(user_node, "e:RegistrationDate", "")
            feedback_score = _safe_find_text(user_node, "e:FeedbackScore", "0")
            reg_addr = user_node.find("e:RegistrationAddress", _NS)
            country = _safe_find_text(reg_addr, "e:Country", "") if reg_addr is not None else ""

            return {
                "registration_date": reg_date,
                "feedback_score": int(feedback_score),
                "location": country,
                "user_id": _safe_find_text(user_node, "e:UserID", ""),
                "status": _safe_find_text(user_node, "e:Status", ""),
            }
        except Exception as e:
            logger.warning("GetUser failed for %s: %s", username, e)
            return {}


async def get_store_details(username: str) -> Dict[str, Any]:
    """
    Fetch store details using GetStore.
    """
    username = _clean_username(username)
    if not username or not EBAY_USER_TOKEN:
        return {}

    headers = _build_headers("GetStore")
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<GetStoreRequest xmlns="urn:ebay:apis:eBLBaseComponents">
  <RequesterCredentials>
    <eBayAuthToken>{EBAY_USER_TOKEN}</eBayAuthToken>
  </RequesterCredentials>
  <UserID>{username}</UserID>
</GetStoreRequest>"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(TRADING_URL, headers=headers, content=body.encode("utf-8"), timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                return {}
            
            root = ET.fromstring(response.text)
            store_node = root.find(".//e:Store", _NS)
            if store_node is None:
                return {}

            return {
                "store_name": _safe_find_text(store_node, "e:Name", ""),
                "logo_url": _safe_find_text(store_node, "e:LogoURL", ""),
                "description": _safe_find_text(store_node, "e:Description", ""),
            }
        except Exception as e:
            logger.warning("GetStore failed for %s: %s", username, e)
            return {}

