import os
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import List, Dict

EBAY_USER_TOKEN = os.getenv("EBAY_USER_TOKEN")
TRADING_URL = "https://api.ebay.com/ws/api.dll"
COMPATIBILITY_LEVEL = "1451"


def fetch_feedback_page(username: str, page: int, per_page: int) -> List[Dict]:
    if not EBAY_USER_TOKEN:
        # meglio fallire esplicitamente: aiuta debugging
        raise RuntimeError("EBAY_USER_TOKEN is not set")

    headers = {
        "X-EBAY-API-CALL-NAME": "GetFeedback",
        "X-EBAY-API-COMPATIBILITY-LEVEL": COMPATIBILITY_LEVEL,
        "X-EBAY-API-SITEID": "101",
        "Content-Type": "text/xml",
    }

    body = f"""<?xml version="1.0" encoding="utf-8"?>
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
    </GetFeedbackRequest>
    """

    response = requests.post(
        TRADING_URL,
        headers=headers,
        data=body,
        timeout=20
    )

    if response.status_code != 200:
        return []

    response.encoding = "utf-8"

    try:
        root = ET.fromstring(response.text)
    except Exception:
        return []

    ns = {"e": "urn:ebay:apis:eBLBaseComponents"}

    feedbacks: List[Dict] = []

    for fb in root.findall(".//e:FeedbackDetail", ns):
        comment = fb.findtext("e:CommentText", default="", namespaces=ns)

        feedbacks.append({
            "user": fb.findtext("e:CommentingUser", default="", namespaces=ns),
            "rating": fb.findtext("e:CommentType", default="", namespaces=ns),
            "comment": comment,
            "time": fb.findtext("e:CommentTime", default="", namespaces=ns),
        })

    return feedbacks


@lru_cache(maxsize=256)
def get_seller_feedback(username: str, limit: int = 200) -> List[Dict]:
    """
    Fetch + cache seller feedback.
    IMPORTANT: no RAG side-effects here (no ingest).
    """
    username = (username or "").strip()
    if not username:
        return []

    per_page = 50
    max_pages = max(1, (limit + per_page - 1) // per_page)

    all_feedback: List[Dict] = []

    for page in range(1, max_pages + 1):
        page_feedback = fetch_feedback_page(username, page, per_page)

        if not page_feedback:
            break

        all_feedback.extend(page_feedback)

        if len(page_feedback) < per_page:
            break

        if len(all_feedback) >= limit:
            break

    return all_feedback[:limit]