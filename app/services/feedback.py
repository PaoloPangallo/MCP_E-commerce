import requests
import xml.etree.ElementTree as ET
import os

EBAY_USER_TOKEN = os.getenv("EBAY_USER_TOKEN")
TRADING_URL = "https://api.ebay.com/ws/api.dll"
COMPATIBILITY_LEVEL = "1451"


def get_seller_feedback(username: str, limit: int = 10):

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
        <EntriesPerPage>{limit}</EntriesPerPage>
        <PageNumber>1</PageNumber>
      </Pagination>
    </GetFeedbackRequest>
    """

    response = requests.post(TRADING_URL, headers=headers, data=body)

    if response.status_code != 200:
        raise RuntimeError("Errore Trading API")

    response.encoding = "utf-8"
    root = ET.fromstring(response.text)
    ns = {"e": "urn:ebay:apis:eBLBaseComponents"}

    feedbacks = []

    for fb in root.findall(".//e:FeedbackDetail", ns):
        feedbacks.append({
            "user": fb.findtext("e:CommentingUser", default="", namespaces=ns),
            "rating": fb.findtext("e:CommentType", default="", namespaces=ns),
            "comment": fb.findtext("e:CommentText", default="", namespaces=ns),
            "time": fb.findtext("e:CommentTime", default="", namespaces=ns),
        })

    return feedbacks