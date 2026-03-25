import urllib.request
import os
from dotenv import load_dotenv

load_dotenv()

EBAY_USER_TOKEN = os.getenv("EBAY_USER_TOKEN")
TRADING_API_URL = "https://api.ebay.com/ws/api.dll"

def run_call(call_name: str, payload: str):
    body = f'<?xml version="1.0" encoding="utf-8"?><{call_name}Request xmlns="urn:ebay:apis:eBLBaseComponents"><RequesterCredentials><eBayAuthToken>{EBAY_USER_TOKEN}</eBayAuthToken></RequesterCredentials><ErrorLanguage>it_IT</ErrorLanguage><WarningLevel>High</WarningLevel>{payload}</{call_name}Request>'.encode("utf-8")
    headers = {
        "X-EBAY-API-COMPATIBILITY-LEVEL": "1271",
        "X-EBAY-API-CALL-NAME": call_name,
        "X-EBAY-API-SITEID": "101",
        "Content-Type": "text/xml",
    }
    req = urllib.request.Request(TRADING_API_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as f:
            return f.read().decode("utf-8")
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    print("--- Download TOTALE Watchlist ---")
    
    # Chiediamo tutto (attivi e finiti)
    payload = """
    <WatchList>
        <Pagination><EntriesPerPage>200</EntriesPerPage></Pagination>
    </WatchList>
    """
    res = run_call("GetMyeBayBuying", payload)
    
    with open("c:/Users/paolo/MCP_ECOM/tmp/watchlist_raw.xml", "w", encoding="utf-8") as f:
        f.write(res)
    
    print(f"Watchlist salvata (Lunghezza caratteri: {len(res)})")
    if "<TotalNumberOfEntries>" in res:
        total = res.split("<TotalNumberOfEntries>")[1].split("</TotalNumberOfEntries>")[0]
        print(f"TOTALE OGGETTI DICHIARATO DA EBAY: {total}")

if __name__ == "__main__":
    main()
