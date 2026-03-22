import logging
import json
from app.mcp.core import mcp, _db_context, resolve_user_by_id

logger = logging.getLogger(__name__)

def _safe_json(data):
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning("JSON serialization failed in resource: %s", exc)
        return '{"status": "error", "message": "serialization failed"}'

@mcp.resource("profile://{session_id}")
async def get_user_profile(session_id: str) -> str:
    """Restituisce il profilo, preferenze e contesto dell'utente loggato"""
    with _db_context() as db:
        user = resolve_user_by_id(session_id)
        if not user:
            return _safe_json({"status": "error", "message": "User not found or session invalid."})
        profile_data = {
            "user_id": user.id,
            "username": getattr(user, "username", "Unknown"),
            "email": getattr(user, "email", "Unknown"),
            "favorite_brands": getattr(user, "favorite_brands", ""),
            "price_preference": getattr(user, "price_preference", ""),
            "language": getattr(user, "language", "it")
        }
        return _safe_json({"status": "ok", "profile": profile_data})

@mcp.resource("ebay://categories")
async def get_ebay_categories() -> str:
    """Restituisce la mappa delle categorie principali eBay con i relativi ID"""
    categories = {
        "Consumer Electronics": {
            "Cell phones & Accessories": "9355",
            "Computers, Tablets & Networking": "58058",
            "Video Games & Consoles": "1249",
            "Cameras & Photo": "625"
        },
        "Fashion": {
            "Clothing, Shoes & Accessories": "11450",
            "Luxury": {
                "Watches, Parts & Accessories": "14324",
                "Handbags & Accessories": "169291"
            }
        },
        "Home & Garden": "11700",
        "Sporting Goods": "888",
        "Toys & Hobbies": "220"
    }
    return _safe_json({"status": "ok", "categories": categories})

@mcp.resource("ebay://market-logic")
async def get_market_logic() -> str:
    """Linee guida per l'interpretazione dei dati di mercato e trend"""
    logic = {
        "price_signals": {
            "downward_trend": "Ottimo momento per comprare, il prezzo sta scendendo rispetto alla media storica.",
            "stable": "Prezzo di mercato standard. Valuta in base alla reputazione del venditore.",
            "spiking": "Domanda alta o scarsità. Potrebbe convenire attendere o cercare varianti meno note."
        },
        "seasonality_tips": [
            "Tech: Sconti forti a Novembre (Black Friday) e fine anno.",
            "Sport: Sconti a fine stagione (es: bici in inverno).",
            "Moda: Saldi stagionali standard."
        ]
    }
    return _safe_json({"status": "ok", "logic": logic})

@mcp.prompt("shopping_expert_prompt")
def shopping_expert() -> str:
    """Restituisce un prompt di base per vestire l'LLM da Shopping Assistant professionale"""
    return (
        "Sei un assistente e-commerce professionale ed etico basato sullo standard MCP. "
        "Il tuo compito è aiutare l'utente a cercare, valutare e confrontare prodotti dal marketplace sfruttando "
        "le tue capacità analitiche. "
        "I tuoi strumenti includono l'accesso diretto alle offerte del giorno e promozioni eBay: usali attivamente quando l'utente cerca risparmio o occasioni. "
        "Evita formattazioni prolisse. Offri prima la tua interpretazione del prodotto ricercato, poi usa "
        "attivamente i search tool. Tieni in fortissima considerazione le letture di Resources asincrone (es: preferenze o profile dell'utente) "
        "per offrire risultati personalizzati. "
        "PUOI USARE IL TOOL 'inspect_mcp_resource(uri=\"...\")' per leggere contenuti aggiuntivi come: "
        "- 'ebay://categories': Mappa categorie e ID ufficiali eBay. "
        "- 'ebay://market-logic': Guida all'interpretazione dei prezzi e trend."
    )

@mcp.prompt("deal_hunter")
def deal_hunter_prompt() -> str:
    """Prompt focalizzato sulla massimizzazione dello sconto e caccia all'affare"""
    return (
        "Sei il 'Deal Hunter' definitivo. Il tuo unico obiettivo è far risparmiare l'utente. "
        "Quando analizzi i prodotti, focalizzati ossessivamente su: "
        "1. Prezzo scontato vs Prezzo originale. "
        "2. Articoli con spedizione gratuita. "
        "3. Venditori che accettano proposte d'acquisto. "
        "4. Confronto tra deals attuali e risultati di ricerca standard per trovare il prezzo minimo assoluto. "
        "Comunica con tono entusiasta ma preciso riguardo ai numeri del risparmio."
    )

@mcp.prompt("tech_expert")
def tech_expert_prompt() -> str:
    """Prompt per assistenza tecnica specialistica su elettronica e informatica"""
    return (
        "Sei un esperto di tecnologia (Tech Guru). Il tuo compito è andare oltre il prezzo. "
        "Analizza le specifiche tecniche (RAM, processore, stato della batteria se indicato, generazione del modello). "
        "Aiuta l'utente a capire se un prodotto 'usato' o 'ricondizionato' è un vero affare tecnico o un rischio. "
        "Usa i tool per confrontare le specifiche tra modelli simili. "
        "Se l'utente non specifica la RAM o lo storage, chiedi chiarimenti per offrire un consiglio tecnico accurato."
    )
