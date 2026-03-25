import logging
from typing import Dict, Any, Annotated, Optional
from pydantic import Field

from app.mcp.core import mcp, _db_context, resolve_user_by_id

logger = logging.getLogger(__name__)


@mcp.tool(
    name="manage_wishlist",
    description=(
        "Gestisce la wishlist dell'utente (lista dei prodotti salvati). "
        "Azioni disponibili: "
        "'add' - aggiunge un prodotto alla wishlist (richiede ebay_id e title); "
        "'remove' - rimuove un prodotto dalla wishlist (richiede ebay_id); "
        "'list' - restituisce tutti i prodotti salvati. "
        "Usa questo tool quando l'utente dice 'salva questo prodotto', 'aggiungi ai preferiti', "
        "'rimuovi dalla wishlist', 'mostrami la mia lista dei desideri', ecc."
    ),
)
async def manage_wishlist(
    session_id: Annotated[str, Field(description="ID di sessione utente")],
    action: Annotated[str, Field(description="Azione da eseguire: 'add', 'remove', o 'list'")],
    ebay_id: Annotated[Optional[str], Field(description="ID eBay del prodotto (richiesto per 'add' e 'remove')")] = None,
    title: Annotated[Optional[str], Field(description="Titolo del prodotto (per 'add')")] = None,
    price: Annotated[Optional[float], Field(description="Prezzo del prodotto (per 'add')")] = None,
    currency: Annotated[Optional[str], Field(description="Valuta del prezzo, es. 'EUR' (per 'add')")] = None,
    condition: Annotated[Optional[str], Field(description="Condizione del prodotto, es. 'Nuovo', 'Usato' (per 'add')")] = None,
    image_url: Annotated[Optional[str], Field(description="URL dell'immagine del prodotto (per 'add')")] = None,
    url: Annotated[Optional[str], Field(description="URL del prodotto su eBay (per 'add')")] = None,
    seller_name: Annotated[Optional[str], Field(description="Nome del venditore (per 'add')")] = None,
) -> Dict[str, Any]:
    """
    Add, remove, or list wishlist items for the current user.
    """
    if not session_id:
        return {"status": "error", "message": "session_id non può essere vuoto."}

    action = (action or "").strip().lower()
    if action not in {"add", "remove", "list"}:
        return {"status": "error", "message": f"Azione non valida: '{action}'. Usa 'add', 'remove', o 'list'."}

    try:
        with _db_context() as db:
            user = resolve_user_by_id(session_id)
            if not user:
                return {"status": "error", "message": "Utente non trovato o session_id non valido."}

            from app.models.wishlist import WishlistItem

            if action == "list":
                items = db.query(WishlistItem).filter(WishlistItem.user_id == user.id).order_by(WishlistItem.added_at.desc()).all()
                item_list = [
                    {
                        "id": item.id,
                        "ebay_id": item.ebay_id,
                        "title": item.title,
                        "price": item.price,
                        "currency": item.currency,
                        "condition": item.condition,
                        "image_url": item.image_url,
                        "url": item.url,
                        "seller_name": item.seller_name,
                    }
                    for item in items
                ]
                return {
                    "status": "ok",
                    "action": "list",
                    "count": len(item_list),
                    "items": item_list,
                    "_backend": "mcp"
                }

            elif action == "add":
                if not ebay_id:
                    return {"status": "error", "message": "ebay_id è richiesto per aggiungere un prodotto."}

                # Prevent duplicates
                existing = db.query(WishlistItem).filter(
                    WishlistItem.user_id == user.id,
                    WishlistItem.ebay_id == ebay_id
                ).first()
                if existing:
                    return {"status": "ok", "message": f"'{title or ebay_id}' è già nella tua wishlist.", "_backend": "mcp"}

                item = WishlistItem(
                    user_id=user.id,
                    ebay_id=ebay_id,
                    title=title,
                    price=price,
                    currency=currency or "EUR",
                    condition=condition,
                    image_url=image_url,
                    url=url,
                    seller_name=seller_name,
                )
                db.add(item)
                db.commit()
                db.refresh(item)

                logger.info("Wishlist item added: user=%s, ebay_id=%s", user.id, ebay_id)

                # Notify resource update for frontend sync
                try:
                    await mcp.notify_resource_updated(f"wishlist://{session_id}")
                except Exception as notify_exc:
                    logger.warning("Could not notify wishlist resource update: %s", notify_exc)

                return {
                    "status": "ok",
                    "action": "add",
                    "message": f"✅ '{title or ebay_id}' aggiunto alla tua wishlist!",
                    "item_id": item.id,
                    "_backend": "mcp"
                }

            elif action == "remove":
                if not ebay_id:
                    return {"status": "error", "message": "ebay_id è richiesto per rimuovere un prodotto."}

                item = db.query(WishlistItem).filter(
                    WishlistItem.user_id == user.id,
                    WishlistItem.ebay_id == ebay_id
                ).first()

                if not item:
                    return {"status": "ok", "message": "Prodotto non trovato nella wishlist.", "_backend": "mcp"}

                db.delete(item)
                db.commit()
                logger.info("Wishlist item removed: user=%s, ebay_id=%s", user.id, ebay_id)

                try:
                    await mcp.notify_resource_updated(f"wishlist://{session_id}")
                except Exception as notify_exc:
                    logger.warning("Could not notify wishlist resource update: %s", notify_exc)

                return {
                    "status": "ok",
                    "action": "remove",
                    "message": f"🗑️ Prodotto rimosso dalla tua wishlist.",
                    "_backend": "mcp"
                }

    except Exception as exc:
        logger.error("Error managing wishlist: %s", exc)
        return {"status": "error", "message": f"Errore interno: {str(exc)}"}
