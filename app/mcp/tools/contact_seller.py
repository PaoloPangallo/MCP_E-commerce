import logging
from typing import Dict, Any, Annotated, Optional
from pydantic import Field

from app.mcp.core import mcp, _tool_error

logger = logging.getLogger(__name__)


@mcp.tool(
    name="contact_seller",
    description=(
        "Genera un link diretto per contattare un venditore eBay tramite la messaggistica ufficiale. "
        "Usa questo tool quando l'utente vuole scrivere al venditore, fare una domanda sull'oggetto, "
        "trattare il prezzo, chiedere informazioni sullo stato, sulla spedizione, ecc. "
        "Restituisce l'URL di contatto da mostrare all'utente come link cliccabile. "
        "Esempi: 'voglio scrivere al venditore', 'come faccio a contattare chi vende questo?', "
        "'voglio chiedere se è disponibile', 'posso trattare il prezzo?'."
    ),
)
async def contact_seller(
    seller_name: Annotated[str, Field(description="Username del venditore su eBay (es. 'jjtech2020')")],
    item_id: Annotated[Optional[str], Field(description="ID numerico dell'oggetto eBay (opzionale, per contestualizzare il messaggio)")] = None,
    session_id: Annotated[str, Field(description="ID di sessione")] = "",
) -> Dict[str, Any]:
    """
    Generates a direct eBay contact link for a seller.
    This is the official way to contact a seller through eBay messaging.
    """
    try:
        if not seller_name or not seller_name.strip():
            return _tool_error(seller_name="", error="seller_name è obbligatorio")

        seller_clean = seller_name.strip()

        # Extract numeric ID if item_id is in v1|...|... format
        final_item_id = None
        if item_id:
            item_clean = item_id.strip()
            if "|" in item_clean:
                parts = item_clean.split("|")
                if len(parts) >= 2:
                    final_item_id = parts[1]
            else:
                final_item_id = item_clean

        # Build the eBay contact URL
        if final_item_id:
            contact_url = (
                f"https://www.ebay.it/cnt/IntermediatedFAQ"
                f"?seller_name={seller_clean}&item_id={final_item_id}"
            )
        else:
            contact_url = (
                f"https://www.ebay.it/cnt/IntermediatedFAQ"
                f"?seller_name={seller_clean}"
            )

        logger.info("Contact seller URL generated | seller=%s | item=%s", seller_clean, final_item_id)

        return {
            "success": True,
            "contact_url": contact_url,
            "seller_name": seller_clean,
            "item_id": final_item_id,
            "message": (
                f"Clicca sul link per inviare un messaggio a **{seller_clean}** "
                f"tramite la messaggistica ufficiale di eBay:\n\n"
                f"👉 [{contact_url}]({contact_url})"
            ),
        }

    except Exception as exc:
        logger.exception("MCP contact_seller failed")
        return _tool_error(seller_name=seller_name, error=str(exc))
