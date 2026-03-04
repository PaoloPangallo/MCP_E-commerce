import requests
import xml.etree.ElementTree as ET
import os

from app.services.rag.vector_store import add_documents
from app.services.rag.bm25_store import add_documents as add_bm25

EBAY_USER_TOKEN = os.getenv("EBAY_USER_TOKEN")
TRADING_URL = "https://api.ebay.com/ws/api.dll"
COMPATIBILITY_LEVEL = "1451"


def fetch_feedback_page(username: str, page: int, per_page: int):

    headers = {
        "X-EBAY-API-CALL-NAME": "GetFeedback",
        "X-EBAY-API-COMPATIBILITY-LEVEL": COMPATIBILITY_LEVEL,
        "X-EBAY-API-SITEID": "101",
        "Content-Type": "text/xml"
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
        timeout=10
    )

    if response.status_code != 200:
        raise RuntimeError("Errore Trading API")

    response.encoding = "utf-8"
    root = ET.fromstring(response.text)

    ns = {"e": "urn:ebay:apis:eBLBaseComponents"}

    feedbacks = []

    for fb in root.findall(".//e:FeedbackDetail", ns):

        comment = fb.findtext("e:CommentText", default="", namespaces=ns)

        feedbacks.append({
            "user": fb.findtext("e:CommentingUser", default="", namespaces=ns),
            "rating": fb.findtext("e:CommentType", default="", namespaces=ns),
            "comment": comment,
            "time": fb.findtext("e:CommentTime", default="", namespaces=ns),
        })

    return feedbacks


def get_seller_feedback(username: str, limit: int = 200):

    per_page = 50
    page = 1

    all_feedback = []

    while len(all_feedback) < limit:

        page_feedback = fetch_feedback_page(username, page, per_page)

        if not page_feedback:
            break

        all_feedback.extend(page_feedback)

        page += 1

        if len(page_feedback) < per_page:
            break

    feedbacks = all_feedback[:limit]

    # ----------------------------
    # RAG indicizzazione (max 30)
    # ----------------------------

    texts = []
    metadata = []

    for f in feedbacks[:30]:

        text = f.get("comment")

        if text and len(text.strip()) > 3:

            clean = text.strip()

            texts.append(clean)

            metadata.append({
                "text": clean,
                "seller": username,
                "type": "seller_feedback"
            })

    if texts:

        try:
            add_documents(texts, metadata)
            add_bm25(texts, metadata)
        except Exception:
            pass

    return feedbacks