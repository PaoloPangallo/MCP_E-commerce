# Valutazione E2E del Tool Calling MCP

## Obiettivo

Valutare il percorso reale dell'agente e-commerce: richiesta utente, pianificazione, catalogo MCP, invocazione del tool, osservazione e chiusura del turno. La valutazione deve produrre un report leggibile e confrontabile fra esecuzioni.

## Ambito

Il nuovo test e' `tests/eval/test_agent_tool_calling_e2e.py`. Esegue `EbayReactAgent.run_stream()` con MCP abilitato e raccoglie gli eventi `tool_start`, `tool_result`, `error` e `final`.

Ogni caso definisce un ID stabile, query, modalita' MCP, intent e tool attesi. Il report JSONL conserva query, intent rilevato, tool pianificati e completati, esito dei tool, errori, risposta finale e latenza.

## Matrice di valutazione

Il file contiene almeno 30 casi, ripartiti fra:

- 8 ricerche prodotto, inclusi budget, brand e query vuota/ambigua;
- 4 confronti e trend di mercato;
- 6 dettagli e spedizioni, con e senza item ID;
- 5 analisi venditore e richieste ibride;
- 3 wishlist e 2 contatti venditore valutati senza eseguire azioni mutanti;
- 2 casi di conversazione;
- 2 casi Playwright, eseguiti solo se la relativa configurazione e' disponibile.

## Criteri di qualita'

Ogni caso ha tre valutazioni indipendenti:

1. `routing_pass`: l'intento o almeno uno dei tool pianificati soddisfa l'aspettativa del caso.
2. `execution_pass`: un tool non mutante atteso restituisce un risultato riuscito; per i casi di sola pianificazione e' `not_applicable`.
3. `completion_pass`: l'agente emette un evento finale senza errori non gestiti e con una risposta non vuota.

Un caso e' `pass` se tutti i criteri applicabili sono superati; e' `skip` se un prerequisito non e' disponibile; e' `fail` negli altri casi. Lo skip e' riportato, mai scambiato per successo.

## Sicurezza e isolamento

- Il test legge soltanto da PostgreSQL, Redis e Qdrant attraverso i normali flussi applicativi.
- I casi `manage_wishlist` e `contact_seller` si fermano alla pianificazione: nessuna modifica alla wishlist e nessun messaggio viene inviato.
- I tool di ricerca possono usare le credenziali eBay gia' configurate; gli errori di rete risultano nel report, non vengono mascherati.
- I casi Playwright richiedono `RUN_PLAYWRIGHT_E2E=1`; altrimenti vengono marcati `skip`.

## Prerequisiti e comandi

Il test richiede `DATABASE_URL`. Redis e Qdrant vengono classificati come disponibili/non disponibili nel report; l'assenza non deve impedire i casi che non ne dipendono. Per una valutazione completa, l'operatore avvia backend e servizi configurati e lancia:

```powershell
$env:RUN_AGENT_TOOL_CALLING_E2E = "1"
python -m pytest tests/eval/test_agent_tool_calling_e2e.py -s -q
```

Senza l'opt-in, l'intero modulo e' saltato per evitare chiamate LLM o di rete accidentali durante la suite ordinaria.

Il report viene scritto in `artifacts/evaluations/agent_tool_calling_e2e.jsonl`; una riga di riepilogo finale mostra pass/fail/skip, tasso di routing, tasso di esecuzione e latenza mediana.
