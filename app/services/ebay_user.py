"""
ebay_user.py
------------
Service layer for eBay operations that require a User Access Token
(as opposed to the server-to-server Client Credentials token used in ebay.py).

Uses the eBay Trading API (SOAP/XML over HTTPS) which is the only API
that exposes Watchlist management for individual buyers.
"""

import logging
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils
from typing import Any, Dict, List, Optional

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_API_URL = (
    "https://api.ebay.com/ws/api.dll"
    if settings.EBAY_ENV == "production"
    else "https://api.sandbox.ebay.com/ws/api.dll"
)

_SITE_ID = "0"  # Global Context
_API_VERSION = "1113"


def _base_headers(call_name: str) -> Dict[str, str]:
    """Build standard Trading API headers."""
    return {
        "X-EBAY-API-CALL-NAME": call_name,
        "X-EBAY-API-SITEID": _SITE_ID,
        "X-EBAY-API-COMPATIBILITY-LEVEL": _API_VERSION,
        "Content-Type": "text/xml; charset=utf-8",
    }


def _xml_body(call_name: str, inner_xml: str) -> str:
    """Wrap inner XML in the standard eBay Trading API envelope with Auth'n'Auth token."""
    token = (settings.EBAY_USER_TOKEN or "").strip()
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<{call_name}Request xmlns="urn:ebay:apis:eBLBaseComponents">'
        f"<Version>{_API_VERSION}</Version>"
        f"<RequesterCredentials><eBayAuthToken>{token}</eBayAuthToken></RequesterCredentials>"
        f"{inner_xml}"
        f"</{call_name}Request>"
    )


def _find_text(root: ET.Element, tag: str, ns: str) -> Optional[str]:
    el = root.find(f"{ns}{tag}")
    return el.text if el is not None else None


def _extract_numeric_id(full_id: str) -> str:
    """
    Extracts the legacy numeric ID from an eBay REST ID.
    Example: 'v1|395631298243|664386198358' -> '395631298243'
    """
    if not full_id:
        return ""
    if "|" in full_id:
        parts = full_id.split("|")
        # Usually the middle part for 'v1|ID|...' or the only part
        if len(parts) >= 2:
            return parts[1]
    return full_id


# ---------------------------------------------------------------------------
# Watchlist — Add
# ---------------------------------------------------------------------------

async def add_to_ebay_watchlist(item_id: str) -> Dict[str, Any]:
    """
    Add an item to the user's eBay Watchlist. 
    Smart logic: if it's a v1|...|... ID, and the first part fails, tries the second part.
    """
    # Potential candidates for ItemID from a REST ID
    candidates = []
    if "|" in item_id:
        parts = item_id.split("|")
        # candidates: Part 1 (v1), Part 2 (usually ItemID), Part 3 (listing or variation)
        if len(parts) >= 2: candidates.append(parts[1])
        if len(parts) >= 3: candidates.append(parts[2])
    else:
        candidates.append(item_id)

    last_error = ""
    for cid in candidates:
        if cid == "0" or not cid: continue
        
        logger.info("Tentativo aggiunta eBay Watchlist | ID: %s", cid)
        body = _xml_body("AddToWatchList", f"<ItemID>{cid}</ItemID>")

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    TRADING_API_URL,
                    headers=_base_headers("AddToWatchList"),
                    content=body.encode("utf-8"),
                )

            root = ET.fromstring(resp.text)
            ns = "{urn:ebay:apis:eBLBaseComponents}"
            ack = (_find_text(root, "Ack", ns) or "").lower()

            if ack in ("success", "warning"):
                watch_count = _find_text(root, "WatchListCount", ns)
                logger.info("eBay Watchlist add OK | ID selezionato: %s | count=%s", cid, watch_count)
                return {"success": True, "watch_count": watch_count, "used_id": cid}
            
            # Extract error
            errors_el = root.find(f"{ns}Errors")
            err_msg = ""
            if errors_el is not None:
                err_code = _find_text(errors_el, "ErrorCode", ns)
                err_msg = _find_text(errors_el, "LongMessage", ns) or ""
                # If error is explicitly 'item not found' or similar, try next candidate
                logger.warning("Fallimento parziale eBay %s | ID: %s | Msg: %s", err_code, cid, err_msg)
                last_error = err_msg
            
            # If item is already in list (20819), we consider it a sort of success or at least we are done
            if "20819" in resp.text:
                logger.info("Oggetto già presente o errore skip | ID: %s", cid)
                # Keep going to try next candidate just in case, but store this info
        
        except Exception as exc:
            logger.error("Eccezione durante sync eBay | ID: %s | %s", cid, exc)
            last_error = str(exc)

    return {"success": False, "message": last_error or "Nessun ID valido trovato per eBay"}


async def send_message_to_seller(item_id: str, body_text: str, question_type: str = "General") -> Dict[str, Any]:
    """
    Sends a message to a seller regarding a specific item.
    Uses AddMemberMessageAAQToSeller Trading API.
    """
    token = (settings.EBAY_USER_TOKEN or "").strip()
    if not token:
        return {"success": False, "message": "EBAY_USER_TOKEN non configurato"}

    # Extract numeric part just in case
    candidates = []
    if "|" in item_id:
        parts = item_id.split("|")
        if len(parts) >= 2: candidates.append(parts[1])
    else:
        candidates.append(item_id)
    
    target_id = candidates[0] if candidates else item_id

    inner_xml = f"""
        <ItemID>{saxutils.escape(target_id)}</ItemID>
        <MemberMessage>
            <Body>{saxutils.escape(body_text)}</Body>
            <DisplayToPublic>false</DisplayToPublic>
            <EmailCopyToSender>true</EmailCopyToSender>
            <QuestionType>{saxutils.escape(question_type)}</QuestionType>
        </MemberMessage>
    """
    
    xml = _xml_body("AddMemberMessageAAQToSeller", inner_xml)
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                TRADING_API_URL,
                headers=_base_headers("AddMemberMessageAAQToSeller"),
                content=xml.encode("utf-8"),
            )

        root = ET.fromstring(resp.text)
        ns = "{urn:ebay:apis:eBLBaseComponents}"
        ack = (_find_text(root, "Ack", ns) or "").lower()

        if ack in ("success", "warning"):
            logger.info("eBay Message sent successfully | Item: %s", target_id)
            return {"success": True}
        
        # Extract error
        errors_el = root.find(f"{ns}Errors")
        err_msg = "Errore sconosciuto da eBay"
        if errors_el is not None:
            err_msg = _find_text(errors_el, "LongMessage", ns) or ""
            
        logger.warning("Fallimento invio messaggio eBay | Msg: %s", err_msg)
        return {"success": False, "message": err_msg}

    except Exception as exc:
        logger.exception("Eccezione durante invio messaggio eBay")
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Watchlist — Remove
# ---------------------------------------------------------------------------

async def remove_from_ebay_watchlist(item_id: str) -> Dict[str, Any]:
    """
    Remove an item from the user's eBay Watchlist via RemoveFromWatchList Trading API.
    """
    numeric_id = _extract_numeric_id(item_id)
    logger.info("Sync eBay Watchlist | REMOVE | Original ID: %s | Numeric ID: %s", item_id, numeric_id)
    
    body = _xml_body(
        "RemoveFromWatchList",
        f"<ItemID>{numeric_id}</ItemID>",
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                TRADING_API_URL,
                headers=_base_headers("RemoveFromWatchList"),
                content=body.encode("utf-8"),
            )
        
        logger.info("eBay RemoveFromWatchList Response | status=%s | body=%s", resp.status_code, resp.text[:400])

        if resp.status_code != 200:
            return {"success": False, "message": f"HTTP {resp.status_code}"}

        root = ET.fromstring(resp.text)
        ns = "{urn:ebay:apis:eBLBaseComponents}"
        ack = _find_text(root, "Ack", ns) or ""

        if ack.lower() in ("success", "warning"):
            logger.info("eBay Watchlist remove OK | item=%s", item_id)
            return {"success": True}

        errors_el = root.find(f"{ns}Errors")
        err_msg = ""
        if errors_el is not None:
            err_msg = _find_text(errors_el, "LongMessage", ns) or ""
        return {"success": False, "message": err_msg or ack}

    except Exception as exc:
        logger.error("eBay RemoveFromWatchList exception | item=%s | %s", item_id, exc)
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Watchlist — Get
# ---------------------------------------------------------------------------

async def get_ebay_watchlist(page_number: int = 1, entries_per_page: int = 25) -> List[Dict[str, Any]]:
    """
    Retrieve the user's eBay Watchlist via GetMyeBayBuying Trading API.
    Returns a list of normalized item dicts.
    """
    inner = (
        "<WatchList>"
        f"<Pagination><EntriesPerPage>{entries_per_page}</EntriesPerPage>"
        f"<PageNumber>{page_number}</PageNumber></Pagination>"
        "</WatchList>"
    )
    body = _xml_body("GetMyeBayBuying", inner)

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                TRADING_API_URL,
                headers=_base_headers("GetMyeBayBuying"),
                content=body.encode("utf-8"),
            )

        logger.info("eBay GetMyeBayBuying Response | status=%s | body=%s", resp.status_code, resp.text[:400])

        if resp.status_code != 200:
            logger.warning("eBay GetMyeBayBuying HTTP %s", resp.status_code)
            return []

        root = ET.fromstring(resp.text)
        ns = "{urn:ebay:apis:eBLBaseComponents}"
        ack = _find_text(root, "Ack", ns) or ""
        if ack.lower() not in ("success", "warning"):
            logger.warning("eBay GetMyeBayBuying failed | ack=%s", ack)
            return []

        items: List[Dict[str, Any]] = []
        watch_list_el = root.find(f"{ns}WatchList")
        if watch_list_el is None:
            return []

        item_array = watch_list_el.find(f"{ns}ItemArray")
        if item_array is None:
            return []

        for item_el in item_array.findall(f"{ns}Item"):
            item_id = _find_text(item_el, "ItemID", ns)
            title = _find_text(item_el, "Title", ns)
            url = _find_text(item_el, "ListingDetails/{urn:ebay:apis:eBLBaseComponents}ViewItemURL", ns)

            # Price
            buying_price_el = item_el.find(f"{ns}BuyItNowPrice") or item_el.find(f"{ns}ConvertedCurrentPrice")
            price = None
            currency = "EUR"
            if buying_price_el is not None:
                try:
                    price = float(buying_price_el.text or "0")
                except ValueError:
                    pass
                currency = buying_price_el.get("currencyID", "EUR")

            # Seller
            seller_el = item_el.find(f"{ns}Seller")
            seller_name = None
            if seller_el is not None:
                seller_name = _find_text(seller_el, "UserID", ns)

            # Picture
            picture_el = item_el.find(f"{ns}PictureDetails")
            image_url = None
            if picture_el is not None:
                image_url = _find_text(picture_el, "GalleryURL", ns)

            condition = _find_text(item_el, "ConditionDisplayName", ns)

            if item_id:
                items.append({
                    "ebay_id": item_id,
                    "title": title,
                    "price": price,
                    "currency": currency,
                    "condition": condition,
                    "image_url": image_url,
                    "url": url,
                    "seller_name": seller_name,
                })

        logger.info("eBay Watchlist fetched | count=%d", len(items))
        return items

    except Exception as exc:
        logger.error("eBay GetMyeBayBuying exception | %s", exc)
        return []


# ---------------------------------------------------------------------------
# User Identity
# ---------------------------------------------------------------------------

async def get_ebay_user_info() -> Optional[Dict[str, Any]]:
    """
    Retrieve basic info about the authenticated eBay user via GetUser.
    Returns {'username': ..., 'feedback_score': ..., 'registration_date': ...} or None.
    """
    body = _xml_body("GetUser", "")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                TRADING_API_URL,
                headers=_base_headers("GetUser"),
                content=body.encode("utf-8"),
            )

        logger.info("eBay GetUser Response | status=%s | body=%s", resp.status_code, resp.text[:400])

        if resp.status_code != 200:
            return None

        root = ET.fromstring(resp.text)
        ns = "{urn:ebay:apis:eBLBaseComponents}"
        ack = _find_text(root, "Ack", ns) or ""
        if ack.lower() not in ("success", "warning"):
            return None

        user_el = root.find(f"{ns}User")
        if user_el is None:
            return None

        return {
            "username": _find_text(user_el, "UserID", ns),
            "feedback_score": _find_text(user_el, "FeedbackScore", ns),
            "registration_date": _find_text(user_el, "RegistrationDate", ns),
            "status": _find_text(user_el, "Status", ns),
            "site": _find_text(user_el, "Site", ns),
        }

    except Exception as exc:
        logger.error("eBay GetUser exception | %s", exc)
        return None
