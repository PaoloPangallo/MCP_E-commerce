from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

EBAY_USER_TOKEN = os.getenv("EBAY_USER_TOKEN")
TRADING_URL = os.getenv("EBAY_TRADING_URL", "https://api.ebay.com/ws/api.dll")
COMPATIBILITY_LEVEL = os.getenv("EBAY_COMPATIBILITY_LEVEL", "1451")
REQUEST_TIMEOUT = int(os.getenv("EBAY_FEEDBACK_TIMEOUT", "10"))
SITE_ID = os.getenv("EBAY_SITE_ID", "101")

_PER_PAGE = 50
_SESSION = requests.Session()

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
        feedbacks.append(
            {
                "user": _safe_find_text(fb, "e:CommentingUser", ""),
                "rating": _safe_find_text(fb, "e:CommentType", ""),
                "comment": _safe_find_text(fb, "e:CommentText", ""),
                "time": _safe_find_text(fb, "e:CommentTime", ""),
            }
        )

    return feedbacks


def fetch_feedback_page(username: str, page: int, per_page: int = _PER_PAGE) -> List[Dict]:
    """
    Fetch a single feedback page from eBay Trading API.
    Returns [] on failure or no results.
    """
    username = _clean_username(username)

    if not username:
        return []

    if not EBAY_USER_TOKEN:
        raise RuntimeError("EBAY_USER_TOKEN is not set")

    headers = _build_headers()
    body = _build_body(username, page, per_page)

    last_error: Optional[Exception] = None

    for attempt in range(2):
        try:
            response = _SESSION.post(
                TRADING_URL,
                headers=headers,
                data=body.encode("utf-8"),
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                logger.warning(
                    "GetFeedback HTTP %s for seller=%s page=%s",
                    response.status_code,
                    username,
                    page,
                )
                return []

            response.encoding = "utf-8"
            return _parse_feedback_page(response.text)

        except requests.Timeout as e:
            last_error = e
            logger.warning(
                "GetFeedback timeout for seller=%s page=%s attempt=%s",
                username,
                page,
                attempt + 1,
            )
        except requests.RequestException as e:
            last_error = e
            logger.warning(
                "GetFeedback request error for seller=%s page=%s attempt=%s: %s",
                username,
                page,
                attempt + 1,
                e,
            )

    if last_error:
        logger.warning(
            "GetFeedback failed after retries for seller=%s page=%s: %s",
            username,
            page,
            last_error,
        )

    return []


@lru_cache(maxsize=256)
def get_seller_feedback(username: str, limit: int = 200) -> List[Dict]:
    """
    Fetch + cache seller feedback.
    No side effects.
    Cache key is based on normalized username + limit.
    """
    username = _clean_username(username)

    if not username:
        return []

    limit = max(1, min(int(limit), 500))
    per_page = min(_PER_PAGE, limit)
    max_pages = max(1, (limit + per_page - 1) // per_page)

    all_feedback: List[Dict] = []

    for page in range(1, max_pages + 1):
        page_feedback = fetch_feedback_page(username, page, per_page)

        if not page_feedback:
            break

        all_feedback.extend(page_feedback)

        if len(all_feedback) >= limit:
            break

        if len(page_feedback) < per_page:
            break

    return all_feedback[:limit]


def clear_feedback_cache() -> None:
    get_seller_feedback.cache_clear()