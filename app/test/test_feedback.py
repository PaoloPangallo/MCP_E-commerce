import requests

EBAY_AUTH_TOKEN = "v^1.1#i^1#I^3#f^0#p^3#r^1#t^Ul4xMF81OkEwRUFBMTNDNDVFRTI4NjU3NUE3MUEyNUI0NUU3NzgyXzJfMSNFXjI2MA=="
SELLER_USERNAME = "bestbuy"   # puoi cambiare
COMPATIBILITY_LEVEL = "1451"

url = "https://api.ebay.com/ws/api.dll"

headers = {
    "X-EBAY-API-CALL-NAME": "GetFeedback",
    "X-EBAY-API-COMPATIBILITY-LEVEL": COMPATIBILITY_LEVEL,
    "X-EBAY-API-SITEID": "101",
    "Content-Type": "text/xml"
}

body = f"""<?xml version="1.0" encoding="utf-8"?>
<GetFeedbackRequest xmlns="urn:ebay:apis:eBLBaseComponents">
  <RequesterCredentials>
    <eBayAuthToken>{EBAY_AUTH_TOKEN}</eBayAuthToken>
  </RequesterCredentials>
  <UserID>{SELLER_USERNAME}</UserID>
  <DetailLevel>ReturnAll</DetailLevel>
  <Pagination>
    <EntriesPerPage>25</EntriesPerPage>
    <PageNumber>1</PageNumber>
  </Pagination>
</GetFeedbackRequest>
"""

response = requests.post(url, headers=headers, data=body)

print(response.status_code)
print(response.text[:2000])