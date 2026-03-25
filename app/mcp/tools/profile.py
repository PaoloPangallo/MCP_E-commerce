import logging
from typing import Dict, Any, Annotated, Optional
from pydantic import Field

from app.mcp.core import mcp, _db_context, resolve_user_by_id

logger = logging.getLogger(__name__)

@mcp.tool(
    name="update_user_preferences",
    description=(
        "Aggiorna le preferenze e il profilo dell'utente loggato nel database. "
        "Usa questo tool proattivamente quando l'utente esprime preferenze esplicite (es. 'Mi piacciono solo le scarpe Nike' -> update_user_preferences(session_id, favorite_brands='Nike')). "
        "Puoi aggiornare brand, budget, istruzioni o tono della conversazione."
    ),
)
async def update_user_preferences(
    session_id: Annotated[str, Field(description="ID di sessione utente")],
    favorite_brands: Annotated[Optional[str], Field(description="Le marche o brand preferiti dell'utente (es. 'Nike, Adidas')")] = None,
    price_preference: Annotated[Optional[str], Field(description="Preferenza di prezzo o budget (es. 'economico', 'Sotto 100€', 'premium')")] = None,
    custom_instructions: Annotated[Optional[str], Field(description="Istruzioni aggiuntive su come vuoi che l'agente interagisca con l'utente")] = None,
    theme: Annotated[Optional[str], Field(description="Tema UI preferito (es. 'light', 'dark')")] = None,
    conversation_tone: Annotated[Optional[str], Field(description="Tono della conversazione (es. 'neutral', 'friendly', 'professional', 'zen')")] = None
) -> Dict[str, Any]:
    """
    Update the user's preferences in the database.
    """
    if not session_id:
        return {"status": "error", "message": "session_id string cannot be empty."}

    try:
        with _db_context() as db:
            user = resolve_user_by_id(session_id)
            if not user:
                return {"status": "error", "message": "Utente non trovato o session_id non valido."}
            
            updates = []
            
            if favorite_brands is not None:
                user.favorite_brands = favorite_brands
                updates.append(f"favorite_brands={favorite_brands}")
                
            if price_preference is not None:
                user.price_preference = price_preference
                updates.append(f"price_preference={price_preference}")
                
            if custom_instructions is not None:
                user.custom_instructions = custom_instructions
                updates.append(f"custom_instructions={custom_instructions}")
                
            if theme is not None:
                user.theme = theme
                updates.append(f"theme={theme}")
                
            if conversation_tone is not None:
                user.conversation_tone = conversation_tone
                updates.append(f"conversation_tone={conversation_tone}")
            
            if updates:
                db.commit()
                logger.info(f"Updated preferences for user {user.id}: {updates}")
                
                # Invalida la risorsa MCP per forzare l'aggiornamento nel client
                try:
                    await mcp.notify_resource_updated(f"profile://{session_id}")
                except Exception as e:
                    logger.warning(f"Could not notify resource update: {e}")
                
            return {
                "status": "ok",
                "message": f"Preferenze aggiornate: {', '.join(updates) if updates else 'Nessuna modifica.'}",
                "updated_fields": updates,
                "_backend": "mcp"
            }
            
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        return {"status": "error", "message": f"Errore interno durente l'aggiornamento: {str(e)}"}
