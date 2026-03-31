from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union, cast

# Approx 4 characters per token as a rough safety heuristic
ROUGH_CHARS_PER_TOKEN = 4
MAX_LLM_PROMPT_TOKENS = 6000


def _estimate_tokens(text: str) -> int:
    return len(str(text)) // ROUGH_CHARS_PER_TOKEN


def _truncate_scratchpad(
    scratchpad: Union[Dict[str, Any], List[Dict[str, Any]]],
    context_budget_chars: int,
) -> str:
    """
    Serializza lo scratchpad come JSON troncando se necessario.
    Accetta sia un Dict singolo (da memory.scratchpad()) sia una List di Dict.
    """
    if not scratchpad:
        return "Nessuna azione precedente."

    # --- Normalizzazione: Dict -> stringa diretta, List -> iterazione ---
    if isinstance(scratchpad, dict):
        full_str = json.dumps(scratchpad, ensure_ascii=False, indent=2, default=str)
        if len(full_str) <= context_budget_chars:
            return full_str
        # Tronca i campi più pesanti (results, feedbacks)
        compact = dict(scratchpad)
        for heavy_key in ("results", "feedbacks", "rag_context", "recent_tool_results"):
            if heavy_key in compact:
                val = compact[heavy_key]
                if isinstance(val, list):
                    # Use a slice that satisfies Pyre's SupportsIndex requirement
                    compact[heavy_key] = [val[i] for i in range(len(val)) if i < 3]
                    compact[f"_{heavy_key}_truncated"] = True
        return json.dumps(compact, ensure_ascii=False, indent=2, default=str)[:context_budget_chars]

    # Lista di Dict
    formatted_items = []
    for item in reversed(scratchpad):
        item_str = json.dumps(item, ensure_ascii=False, indent=2, default=str)
        formatted_items.insert(0, item_str)

    full_str = json.dumps(
        [json.loads(i) for i in formatted_items], ensure_ascii=False, indent=2
    )

    if len(full_str) <= context_budget_chars:
        return full_str

    while (
        len(formatted_items) > 1
        and len(
            json.dumps([json.loads(i) for i in formatted_items])
        ) > context_budget_chars
    ):
        formatted_items.pop(0)

    leftover = json.dumps(
        [json.loads(i) for i in formatted_items], ensure_ascii=False, indent=2
    )
    return f"[... {len(scratchpad) - len(formatted_items)} older items omitted ...]\n" + leftover


PLANNER_SYSTEM_PROMPT = """
Sei ebayGPT, il cervello strategico di un assistente e-commerce avanzato. Il tuo compito è decidere la PROSSIMA AZIONE MIGLIORE.

REGOLE DI PIANIFICAZIONE:
1. **Priorità alla Ricerca**: Se l'utente esprime un interesse per un prodotto, marca o categoria, DEVI usare `search_products`. Non chattare a vuoto se puoi cercare.
2. **Gestione del Contesto (CRITICO)**: Se l'utente usa pronomi o riferimenti impliciti (es: "ne", "quello", "neri?", "più economico"), DEVI guardare `session_memory` per capire di cosa si sta parlando e ricostruire una query COMPLETA per `search_products`.
3. **Analisi del Venditore**: Se l'utente chiede se può fidarsi o come sono i feedback, usa `analyze_seller`.
4. **Approccio Graduale**: Non chiamare troppi tool insieme. Risolvi prima il bisogno principale (trovare il prodotto) e poi approfondisci.
5. **Investigazione Proattiva (CRITICO)**: Se i risultati della ricerca sono ambigui, DEVI usare `get_item_details` sull'`item_id` del candidato più promettente per leggere la descrizione completa e gli aspetti tecnici PRIMA di dare il verdetto.
6. **Pensiero in Italiano**: Tutti i tuoi ragionamenti nel campo `thought` devono essere in ITALIANO.
7. **JSON Rigido**: Rispondi SOLO con un JSON valido che segua lo schema richiesto.
8. **Confronto Efficiente (CRITICO)**: Se l'utente chiede un confronto tra prodotti che sono GIÀ stati cercati o sono presenti nei `top_results` del scratchpad, NON chiamare `search_products`. Usa direttamente `comparison` passando i nomi dei modelli specifici. Evita di ripetere ricerche se hai già i dati.
9. **Analisi Vision (CRITICO)**: Se vedi una `vision_description` nel contesto, usala come fonte primaria per i dettagli dell'oggetto. Se la query utente è generica ma hai una descrizione dettagliata dall'immagine, dai priorità ai dettagli dell'immagine per la ricerca.
10. **Analisi di Mercato (CRITICO)**: Se l'utente chiede trend, prezzi medi sul web, popolarità o interesse nel tempo, DEVI usare `market_trends`. Non confonderlo con la semplice ricerca eBay.
11. **Contatto Venditore (CRITICO)**: Se l'utente vuole scrivere al venditore, chiedere informazioni, trattare il prezzo o contattare chi vende un oggetto, DEVI usare `contact_seller` passando `seller_name` e, se disponibile, `item_id`.
12. **Analisi Venditore Target**: Se l'utente menziona un venditore specifico (es: "venduto da pegaso_italia", "di monclick"), DEVI pianificare `analyze_seller` per quel venditore per darne un profilo di affidabilità completo.

POLICY STRUMENTI:
- `search_products`: Discovery, shopping, prezzi.
- `analyze_seller`: Affidabilità, reputazione, trust.
- `get_item_details`: Specifiche tecniche profonde di un `item_id` già trovato.
- `get_shipping_costs`: Costi di spedizione esatti (serve CAP).
- `market_trends`: Prezzi medi sul web e trend di interesse (usa Google Shopping e Trends tramite SerpApi).
- `get_ebay_deals`: Recupera le migliori offerte, sconti e promozioni a tempo limitato da eBay. Usalo quando l'utente cerca risparmio o occasioni. Se il messaggio dell'utente contiene un ID categoria esplicito (es: "(ID: 9355)"), DEVI passarlo come parametro `category_id` al tool.
- `ebay_scrape`: Ricerca avanzata tramite browser (Playwright). Usalo **MANDATORIAMENTE** se l'utente menziona "Playwright", "mostra browser", "browser reale" o "visibile". Se l'utente scrive "MODALITÀ VISIBILE", imposta il parametro `visible=true`.
- `manage_wishlist`: Gestisce i prodotti salvati dell'utente. Usalo per aggiungere, rimuovere o elencare i preferiti (wishlist).
- `contact_seller`: Genera un link diretto per contattare un venditore eBay. Usalo quando l'utente vuole scrivere al venditore, fare domande sull'oggetto, trattare il prezzo o chiedere disponibilità. Richiede `seller_name` e opzionalmente `item_id`.
- `conversation`: SOLO se la richiesta è puramente chiacchiericcio senza alcun intento di acquisto o ricerca.

SCHEMA DI USCITA:
{
  "thought": "Spiega brevemente la tua strategia in ITALIANO",
  "intent": "conversation|seller_analysis|product_search|hybrid|comparison|item_details|shipping|market_trends|deals|wishlist|contact_seller|playwright_search",
  "action": "tool_name|finish",
  "action_input": {},
  "final_answer": null
}
""".strip()


CONVERSATION_ANSWER_SYSTEM_PROMPT = """
Sei ebayGPT, un assistente e-commerce amichevole e competente.
Rispondi in modo naturale, colloquiale e conciso – come farebbe un esperto di shopping in carne e ossa con un amico.

REGOLE:
- Tono caldo, diretto e personale. Usa "tu" informale.
- Rispondi in Italiano (o nella lingua dell'utente).
- Per saluti o domande generali: 1-3 frasi, no elenchi, no titoli markdown.
- Per domande su eBay/shopping: dai informazioni utili, offriti di cercare se serve.
- MAI "Come posso aiutarti oggi?" come unica risposta – aggiungi sempre valore.
- Se hai storico di conversazione, usalo per dare risposte contestualizzate.
""".strip()


PLAYWRIGHT_WORLD_SYSTEM_PROMPT = """
### [MODALITÀ BROWSER PLAYWRIGHT ATTIVA]
Sei in modalità Browser Playwright. Hai accesso a un browser Chromium REALE e VISIBILE sullo schermo.

TOOL DISPONIBILI IN QUESTA MODALITÀ:
- `search_products`: Cerca prodotti su eBay con scraping visivo tramite browser reale. Il browser si apre sullo schermo.
- `contact_seller_playwright`: Contatta direttamente un venditore eBay inviando un messaggio dalla pagina del prodotto.

FLUSSO STANDARD:
1. Usa `search_products` per cercare i prodotti (il browser si aprirà visibile).
2. L'utente sceglie il prodotto di interesse.
3. Usa `contact_seller_playwright` con `product_url` (URL del prodotto scelto dai top_results) e `message` (testo del messaggio).

REGOLE CRITICHE:
- Per `contact_seller_playwright` estrai `product_url` dal campo `url` dei `top_results` nel scratchpad.
- Se l'utente specifica già un messaggio, usalo verbatim nel parametro `message`.
- Se `contact_seller_playwright` ritorna `contact_status=login_required`, informa l'utente che deve effettuare il login su eBay nel browser aperto.
- NON usare `analyze_seller`, `get_ebay_deals` o altri tool non presenti in questa modalità.
### [FINE MODALITÀ PLAYWRIGHT]
""".strip()


FINAL_ANSWER_SYSTEM_PROMPT = """
Sei ebayGPT, il consulente esperto di shopping ufficiale. Il tuo obiettivo è guidare l'utente verso l'acquisto migliore, agendo come un personal shopper tecnico e appassionato.

TONO E PERSONA:
- Sei autorevole, cortese e profondamente competente.
- Comunica in Italiano in modo naturale.
- Usa uno stile consulenziale: non limitarti a elencare, ma consiglia e giustifica.
{TONE_PLACEHOLDER}

STRUTTURA DELLA RISPOSTA (SEGUI QUESTO TEMPLATE):
```
## Analisi

[Inserisci qui l'analisi del contesto e della ricerca...]

[Tabella prodotti se presenti]

## Affidabilità

[Inserisci qui l'analisi del trust score e dei venditori...]

## Verdetto

[Inserisci qui il tuo consiglio finale...]
```

REGOLE DI FORMATTAZIONE E STILE (CRITICO):
- **DOPPIO INVIO**: Dopo ogni titolo (## Titolo) DEVI inserire DUE INVIO (riga vuota). Se non lasci la riga vuota, il sistema non leggerà correttamente la formattazione.
- **NO ATTACCATO**: Non scrivere mai il testo subito dopo il titolo sulla stessa riga.
- **MARGINE**: Lascia molto spazio tra le sezioni ## Analisi, ## Affidabilità e ## Verdetto.
- **NO HALLUCINATION FILTRI**: Se la ricerca restituisce 0 risultati, NON inventare che l'utente ha usato filtri come "nuovo" o "massima RAM".
- **SINTESI FINALE**: È FONDAMENTALE che dopo la tabella (o i blocchi offerte) tu scriva SEMPRE le frasi conclusive nei paragrafi ## Affidabilità e ## Verdetto. NON fermarti all'elenco dati!
- **FILTRI NEGATIVI (CRITICO)**: Se l'utente specifica di NON volere qualcosa (es. "senza cappuccio", "no nero", "nè zip"), DEVI assicurarti che i prodotti selezionati per la tabella e l'analisi rispettino RIGOROSAMENTE queste esclusioni. NON proporre mai articoli che contengano attributi esplicitamente vietati.
- **TABELLE**: Usa un'unica tabella completa. NON spezzare mai la tabella in più parti. La colonna "Link" deve essere l'ULTIMA a destra. (IMPORTANTE: Usa la tabella SOLO se hai risultati concreti).
- **ESEMPIO TABELLA**:
  | Prodotto | Dettaglio 1 | Dettaglio 2 | Condizione | Prezzo | Venditore | Trust | Link |
  |---|---|---|---|---|---|---|---|
  | Nome... | Testo... | Testo... | Usato | 400€ | Nome | 90% | [Vedi](url) |
- **URL OBBLIGATORI**: Usa sempre il campo `url` nella colonna Link senza inventare nulla.

VINCOLI:
- NON inventare mai prodotti, prezzi o link. Usa solo i dati forniti.
- Sii DIMOSTRATIVO: spiega e consiglia con l'entusiasmo della tua nuova personalità!
- Nelle richieste puramente conversazionali (es. "ciao", "come va?"), IGNORA il template "Analisi/Affidabilità/Verdetto" e rispondi in modo SUPER DISCORSIVO, amichevole e sintetico senza generare tabelle o formattazioni.
""".strip()


def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def _compact_tool_catalog_for_prompt(tool_catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    compact: Dict[str, Dict[str, Any]] = {}
    for name, spec in (tool_catalog or {}).items():
        if not isinstance(spec, dict):
            continue

        examples = spec.get("examples") or []
        compact[name] = {
            "description": spec.get("description"),
            "input_schema": spec.get("input_schema"),
            "required_fields": spec.get("required_fields") or [],
            "tags": spec.get("tags") or [],
            "examples": [examples[i] for i in range(len(examples)) if i < 2] if isinstance(examples, list) else [],
            "state_key": spec.get("state_key"),
            "cost": spec.get("cost"),
            "latency_class": spec.get("latency_class"),
            "dependencies": spec.get("dependencies") or [],
            "produced_entities": spec.get("produced_entities") or [],
        }
    return compact


def _compact_scratchpad_for_prompt(scratchpad: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "steps_done": scratchpad.get("steps_done"),
        "intent": scratchpad.get("intent"),
        "pending_tasks": scratchpad.get("pending_tasks") or [],
        "search_status": scratchpad.get("search_status"),
        "seller_status": scratchpad.get("seller_status"),
        "has_search_payload": scratchpad.get("has_search_payload"),
        "has_seller_payload": scratchpad.get("has_seller_payload"),
        "has_search_results": scratchpad.get("has_search_results"),
        "last_seller_name": scratchpad.get("last_seller_name"),
        "top_results": scratchpad.get("top_results") or [],
        "deals": scratchpad.get("deals"),
        "seller_summary": scratchpad.get("seller_summary"),
        "search_analysis": scratchpad.get("search_analysis"),
        "metrics": scratchpad.get("metrics"),
        "tool_calls": scratchpad.get("tool_calls") or {},
        "llm_calls": scratchpad.get("llm_calls") or {},
        "tool_states": scratchpad.get("tool_states") or {},
        "recent_observations": scratchpad.get("recent_observations") or [],
        "recent_errors": scratchpad.get("recent_errors") or [],
        "session_memory": scratchpad.get("session_memory") or {},
        "long_term_memory": scratchpad.get("long_term_memory") or {},
    }


def _compact_final_data_for_prompt(final_data: Dict[str, Any]) -> Dict[str, Any]:
    search = final_data.get("search") or {}
    seller = final_data.get("seller") or {}

    search_data = search.get("results") or []
    compact_search = {
        "results_count": search.get("results_count"),
        "ebay_query_used": search.get("ebay_query_used"),
        "analysis": search.get("analysis"),
        "metrics": search.get("metrics") or search.get("ir_metrics"),
        "top_results": [search_data[i] for i in range(len(search_data)) if i < 6] if isinstance(search_data, list) else [],
    } if search else None

    compact_seller = {
        "seller_name": seller.get("seller_name"),
        "count": seller.get("count"),
        "trust_score": seller.get("trust_score"),
        "sentiment_score": seller.get("sentiment_score"),
        "error": seller.get("error"),
    } if seller else None

    tool_states_compact = {}
    for k, v in (final_data.get("tool_states") or {}).items():
        if isinstance(v, dict):
            # Rimuoviamo il campo "data" o "results" per evitare payload mostruosi (causa errore HTTP 400 da Ollama)
            compact_v = {key: val for key, val in v.items() if key != "data"}
            tool_states_compact[k] = compact_v
        else:
            tool_states_compact[k] = v

    return {
        "intent": final_data.get("intent"),
        "search": compact_search,
        "deals": final_data.get("deals"),
        "seller": compact_seller,
        "compare": final_data.get("compare"),
        "top_result": final_data.get("top_result"),
        "last_seller_name": final_data.get("last_seller_name"),
        "search_analysis": final_data.get("search_analysis"),
        "metrics": final_data.get("metrics"),
        "errors": final_data.get("errors") or [],
        "pending_tasks": final_data.get("pending_tasks") or [],
        "tool_states_summary": tool_states_compact,
        "recent_observations": final_data.get("recent_observations") or [],
        "session_memory": final_data.get("session_memory") or {},
        "long_term_memory": final_data.get("long_term_memory") or {},
    }


def build_planner_prompt(
    user_query: str,
    scratchpad: Union[Dict[str, Any], List[Dict[str, Any]]],
    step_index: int,
    max_steps: int,
    tool_catalog: Dict[str, Dict[str, Any]],
    custom_instructions: Optional[str] = None,
    tone: Optional[str] = None,
    mcp_mode: str = "standard",
) -> str:
    compact_tool_catalog = _compact_tool_catalog_for_prompt(tool_catalog)
    tools_json = json.dumps(compact_tool_catalog, ensure_ascii=False, indent=2)

    system_prompt = PLANNER_SYSTEM_PROMPT

    # Playwright world: inject specialized prompt at the top
    if mcp_mode == "playwright_browser":
        system_prompt = PLAYWRIGHT_WORLD_SYSTEM_PROMPT + "\n\n" + system_prompt

    if custom_instructions:
        # INIEZIONE AD ALTA PRIORITÀ: Sopra le regole standard del planner
        system_prompt = (
            f"### [ISTRUZIONI PERSONALIZZATE UTENTE - PRIORITÀ MASSIMA]\n"
            f"{custom_instructions}\n"
            f"### [FINE ISTRUZIONI PERSONALIZZATE]\n\n"
            f"{PLANNER_SYSTEM_PROMPT}"
        )

    if tone:
        tone_instruction = f"\nTONO RICHIESTO: {tone.upper()}. Mantieni questo stile in tutte le tue interazioni.\n"
        system_prompt = tone_instruction + system_prompt

    # Estimate token usage to safeguard context window
    current_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(tools_json) + _estimate_tokens(user_query)
    budget_chars = max(1000, (MAX_LLM_PROMPT_TOKENS - current_tokens) * ROUGH_CHARS_PER_TOKEN)

    scratchpad_str = _truncate_scratchpad(scratchpad, budget_chars)

    vision_desc = scratchpad.get("vision_description") if isinstance(scratchpad, dict) else None
    vision_context = f"Vision description: {vision_desc}\n" if vision_desc else ""

    return (
        f"{system_prompt}\n\n"
        f"Step:{step_index}/{max_steps}\n"
        f"Available tools:{tools_json}\n"
        f"{vision_context}"
        f"User query:{user_query}\n"
        f"Scratchpad:{scratchpad_str}\n"
        f"Return JSON only."
    )


def build_final_answer_prompt(
    user_query: str,
    scratchpad: Union[Dict[str, Any], List[Dict[str, Any]]],
    final_data: Dict[str, Any],
    custom_instructions: Optional[str] = None,
    tone: Optional[str] = None,
) -> str:
    compact_final_data = _compact_final_data_for_prompt(final_data)
    context_info = _compact_json(compact_final_data)

    system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
    if custom_instructions:
        # INIEZIONE AD ALTA PRIORITÀ: Le istruzioni utente devono sovrascrivere il tono di default
        system_prompt = (
            f"### [REGOLE DI RISPOSTA UTENTE - PRIORITÀ ASSOLUTA]\n"
            f"DEVI rispecchiare rigorosamente queste istruzioni nello stile e nel contenuto:\n"
            f"{custom_instructions}\n"
            f"### [FINE REGOLE UTENTE]\n\n"
            f"{FINAL_ANSWER_SYSTEM_PROMPT}"
        )

    if tone:
        tone_desc = {
            "amichevole": "Usa un tono molto amichevole, caloroso e informale. Usa emoji e sii entusiasta.",
            "professionale": "Usa un tono formale, asciutto e professionale. Evita troppi fronzoli e sii preciso.",
            "neutral": "Usa un tono equilibrato, educato e naturale."
        }.get(tone.lower(), "Usa un tono naturale.")
        system_prompt = system_prompt.replace("{TONE_PLACEHOLDER}", f"- {tone_desc}")
    else:
        system_prompt = system_prompt.replace("{TONE_PLACEHOLDER}", "")
        
    pref_str = ""
    # Estrai le preferenze utente dalla long_term_memory nello scratchpad
    ltm = compact_final_data.get("long_term_memory") or {}
    if ltm:
        prefs = ltm.get("user_preferences") or {}
        behaviour = ltm.get("user_behaviour") or {}
        pref_parts: List[str] = []

        # ── Brand preferences (DB is ground truth) ──────────────────────
        db_brands = prefs.get("db_favorite_brands")
        if db_brands:
            pref_parts.append(f"Brand preferiti (confermati): {db_brands}")

        # Validated brands from parser (more reliable than legacy brand_hints)
        validated_brands = prefs.get("validated_brands") or []
        if isinstance(validated_brands, list) and validated_brands:
            pref_parts.append(f"Brand cercati di recente: {', '.join(str(b) for b in validated_brands[:3])}")
        elif prefs.get("recent_brand_hints"):
            hints = prefs.get("recent_brand_hints") or []
            if isinstance(hints, list):
                pref_parts.append(f"Interessi recenti: {', '.join(str(b) for b in hints[:3])}")

        # ── Price / budget ────────────────────────────────────────────────
        db_price = prefs.get("db_price_preference")
        if db_price:
            pref_parts.append(f"Budget medio abituale: ~{db_price}\u20ac")

        # ── Condition preference ──────────────────────────────────────────
        db_condition = prefs.get("db_condition_preference")
        if db_condition:
            cond_label = {"new": "nuovo", "used": "usato", "refurbished": "ricondizionato"}.get(db_condition, db_condition)
            pref_parts.append(f"Condizione preferita: {cond_label}")

        # ── Category affinities ───────────────────────────────────────────
        category_history = prefs.get("category_history") or []
        if isinstance(category_history, list) and category_history:
            pref_parts.append(f"Categorie di interesse: {', '.join(str(c) for c in category_history[:3])}")

        # ── Interaction depth ─────────────────────────────────────────────
        depth = prefs.get("db_interaction_depth")
        if depth and depth != "browser":
            depth_label = {"researcher": "utente approfondito", "power_buyer": "acquirente esperto"}.get(depth, depth)
            pref_parts.append(f"Profilo comportamentale: {depth_label}")

        # ── Sellers ───────────────────────────────────────────────────────
        if prefs.get("favorite_sellers"):
            sellers = prefs.get("favorite_sellers") or []
            if isinstance(sellers, list):
                pref_parts.append(f"Venditori preferiti: {', '.join(str(s) for s in sellers[:3])}")

        # ── Previous searches ─────────────────────────────────────────────
        prev = ltm.get("previous_searches") or []
        if isinstance(prev, list) and prev:
            pref_parts.append(f"Ricerche precedenti: {', '.join(str(q) for q in prev[:3])}")

        if pref_parts:
            pref_str = "\n".join(pref_parts)

    # Make sure we don't blow up context with massive scratchpad
    base_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_query) + _estimate_tokens(context_info)
    budget_chars = max(1000, (MAX_LLM_PROMPT_TOKENS - base_tokens) * ROUGH_CHARS_PER_TOKEN)
    
    scratch_str = _truncate_scratchpad(scratchpad, budget_chars)

    # Inject explicit results count so the LLM can't ignore them
    results_count = 0
    if isinstance(compact_final_data.get("search"), dict):
        results_count = compact_final_data["search"].get("results_count") or len(compact_final_data["search"].get("top_results") or [])

    results_hint = ""
    if results_count > 0:
        ebay_q = (compact_final_data.get("search") or {}).get("ebay_query_used") or "N/A"
        results_hint = f"\nSEARCH_RESULTS_AVAILABLE: {results_count} products found (eBay query: '{ebay_q}'). You MUST present these results to the user.\n"

    return f"""{system_prompt}
{results_hint}
CONVERSATION CONTEXT:
{context_info}

USER STATUS/PREFERENCES:
{pref_str}

OBSERVATIONS HISTORY:
{scratch_str}

USER REQUEST:
{user_query}

Write the final answer now.
"""