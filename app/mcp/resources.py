import json
import logging
from app.mcp.core import mcp, _db_context, resolve_user_by_id
from app.services.user_profiling import build_profile_context

logger = logging.getLogger(__name__)


def _safe_json(data):
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON serialization failed in resource: %s", exc)
        return '{"status": "error", "message": "serialization failed"}'


@mcp.resource("profile://{session_id}")
async def get_user_profile(session_id: str) -> str:
    """Restituisce il profilo completo, preferenze e contesto dell'utente loggato"""
    with _db_context():
        user = resolve_user_by_id(session_id)
        if not user:
            return _safe_json({"status": "error", "message": "User not found or session invalid."})

        # Base fields
        profile_data = {
            "user_id": user.id,
            "username": getattr(user, "username", "Unknown"),
            "email": getattr(user, "email", "Unknown"),
            "favorite_brands": getattr(user, "favorite_brands", ""),
            "price_preference": getattr(user, "price_preference", ""),
            "custom_instructions": getattr(user, "custom_instructions", None),
            "conversation_tone": getattr(user, "conversation_tone", "neutral"),
            "theme": getattr(user, "theme", "light"),
            "language": getattr(user, "language", "it"),
        }

        # Auto-learned profiling fields (new, retrocompatible)
        try:
            profile_ctx = build_profile_context(user)
            profile_data["condition_preference"] = profile_ctx.get("condition_preference")
            profile_data["interaction_depth"] = profile_ctx.get("interaction_depth") or "browser"
            profile_data["category_affinities"] = profile_ctx.get("category_affinities") or []
            profile_data["contextual_budgets"] = profile_ctx.get("contextual_budgets") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enrich profile with auto-learned fields: %s", exc)
            profile_data["condition_preference"] = None
            profile_data["interaction_depth"] = "browser"
            profile_data["category_affinities"] = []
            profile_data["contextual_budgets"] = {}

        return _safe_json({"status": "ok", "profile": profile_data})


@mcp.prompt("shopping_expert_prompt")
def shopping_expert() -> str:
    """Restituisce un prompt di base per vestire l'LLM da Shopping Assistant professionale"""
    return (
        "Sei un assistente e-commerce professionale ed etico basato sullo standard MCP. "
        "Il tuo compito è aiutare l'utente a cercare, valutare e confrontare prodotti dal marketplace sfruttando "
        "I tuoi strumenti includono l'accesso diretto alle offerte del giorno e promozioni eBay: usali attivamente quando l'utente cerca risparmio o occasioni. "
        "Evita formattazioni prolisse. Offri prima la tua interpretazione del prodotto ricercato, poi usa "
        "attivamente i search tool. Tieni in fortissima considerazione le letture di Resources asincrone (es: preferenze o profile dell'utente) "
        "per offrire risultati personalizzati."
    )
