from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

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
                if isinstance(val, list) and len(val) > 3:
                    compact[heavy_key] = val[:3]
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

POLICY STRUMENTI:
- `search_products`: Discovery, shopping, prezzi.
- `analyze_seller`: Affidabilità, reputazione, trust.
- `get_item_details`: Specifiche tecniche profonde di un `item_id` già trovato.
- `get_shipping_costs`: Costi di spedizione esatti (serve CAP).
- `get_marketplace_metadata`: Politiche eBay (condizioni, resi).
- `conversation`: SOLO se la richiesta è puramente chiacchiericcio senza alcun intento di acquisto o ricerca.

SCHEMA DI USCITA:
{
  "thought": "Spiega brevemente la tua strategia in ITALIANO",
  "intent": "conversation|seller_analysis|product_search|hybrid|comparison|item_details|shipping|metadata",
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


FINAL_ANSWER_SYSTEM_PROMPT = """
Sei ebayGPT, il consulente esperto di shopping ufficiale. Il tuo obiettivo è guidare l'utente verso l'acquisto migliore, agendo come un personal shopper tecnico e appassionato.

TONO E PERSONA:
- Sei autorevole, cortese e profondamente competente.
- Comunica in Italiano in modo naturale.
- Usa uno stile consulenziale: non limitarti a elencare, ma consiglia e giustifica.

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
- **STILE SUPER-LOQUACE E UMANO**: Parla SEMPRE a ruota libera, come un simpatico, logorroico, ma espertissimo personal shopper in carne e ossa. Usa espressioni ricche, entusiaste e colloquiali (es. "Ho spulciato tutto il catalogo per te e non immagini cosa ho scovato!", "Mettiti comodo perché ho analizzato i dettagli e abbiamo delle opzioni pazzesche", ecc.).
- **NASCONDI I DETTAGLI TECNICI**: NON dire MAI frasi robotiche come "Ho effettuato una ricerca", "Il motore ha restituito", "Il sistema mi dice", "I parametri della tua query". Fai finta di essere tu ad aver guardato le vetrine con i tuoi occhi.
- **DOPPIO INVIO**: Dopo ogni titolo (## Titolo) DEVI inserire DUE INVIO (riga vuota). Se non lasci la riga vuota, il sistema non leggerà correttamente la formattazione.
- **NO ATTACCATO**: Non scrivere mai il testo subito dopo il titolo sulla stessa riga.
- **MARGINE**: Lascia molto spazio tra le sezioni ## Analisi, ## Affidabilità e ## Verdetto.
- **NO HALLUCINATION FILTRI**: Se la ricerca restituisce 0 risultati, NON inventare che l'utente ha usato filtri come "nuovo" o "massima RAM".
- **SINTESI FINALE**: È FONDAMENTALE che dopo la tabella tu scriva SEMPRE le frasi conclusive nei paragrafi ## Affidabilità e ## Verdetto. NON fermarti alla tabella!
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

        compact[name] = {
            "description": spec.get("description"),
            "input_schema": spec.get("input_schema"),
            "required_fields": spec.get("required_fields") or [],
            "tags": spec.get("tags") or [],
            "examples": (spec.get("examples") or [])[:2],
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
        "has_metadata_payload": scratchpad.get("has_metadata_payload"),
        "last_seller_name": scratchpad.get("last_seller_name"),
        "top_results": scratchpad.get("top_results") or [],
        "metadata": scratchpad.get("metadata"),
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

    compact_search = {
        "results_count": search.get("results_count"),
        "analysis": search.get("analysis"),
        "metrics": search.get("metrics") or search.get("ir_metrics"),
        "top_results": (search.get("results") or [])[:3],
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
        "metadata": final_data.get("metadata"),
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
) -> str:
    compact_tool_catalog = _compact_tool_catalog_for_prompt(tool_catalog)
    tools_json = json.dumps(compact_tool_catalog, ensure_ascii=False, indent=2)

    system_prompt = PLANNER_SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += f"\n\nUSER CUSTOM INSTRUCTIONS:\n{custom_instructions}"

    # Estimate token usage to safeguard context window
    current_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(tools_json) + _estimate_tokens(user_query)
    budget_chars = max(1000, (MAX_LLM_PROMPT_TOKENS - current_tokens) * ROUGH_CHARS_PER_TOKEN)

    scratchpad_str = _truncate_scratchpad(scratchpad, budget_chars)

    return (
        f"{system_prompt}\n\n"
        f"Step:{step_index}/{max_steps}\n"
        f"Available tools:{tools_json}\n"
        f"User query:{user_query}\n"
        f"Scratchpad:{scratchpad_str}\n"
        f"Return JSON only."
    )


def build_final_answer_prompt(
    user_query: str,
    scratchpad: Union[Dict[str, Any], List[Dict[str, Any]]],
    final_data: Dict[str, Any],
    custom_instructions: Optional[str] = None,
) -> str:
    compact_final_data = _compact_final_data_for_prompt(final_data)
    context_info = _compact_json(compact_final_data)

    system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += f"\n\nUSER CUSTOM INSTRUCTIONS (Apply these specifically to your tone/style/content):\n{custom_instructions}"
        
    pref_str = ""
    # Estrai le preferenze utente dalla long_term_memory nello scratchpad
    # (build_final_answer_prompt riceve final_data che include long_term_memory)
    ltm = compact_final_data.get("long_term_memory") or {}
    if ltm:
        prefs = ltm.get("user_preferences") or {}
        pref_parts = []
        if prefs.get("favorite_sellers"):
            pref_parts.append(f"Venditori preferiti: {', '.join(str(s) for s in prefs['favorite_sellers'][:3])}")
        if prefs.get("recent_brand_hints"):
            pref_parts.append(f"Brand/hint recenti: {', '.join(str(b) for b in prefs['recent_brand_hints'][:3])}")
        prev = ltm.get("previous_searches") or []
        if prev:
            pref_parts.append(f"Ricerche precedenti: {', '.join(str(q) for q in prev[:3])}")
        if pref_parts:
            pref_str = "\n".join(pref_parts)

    # Make sure we don't blow up context with massive scratchpad
    base_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_query) + _estimate_tokens(context_info)
    budget_chars = max(1000, (MAX_LLM_PROMPT_TOKENS - base_tokens) * ROUGH_CHARS_PER_TOKEN)
    
    scratch_str = _truncate_scratchpad(scratchpad, budget_chars)

    return f"""{system_prompt}

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