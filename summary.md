# Evaluation Framework - Status & Next Steps

Questo documento riassume l'infrastruttura di test e benchmarking (NLU & RAG) preparata per il progetto **MCP E-Commerce**, fornendo contesto ed istruzioni per le prossime fasi di sviluppo e QA. Da fornire in input agli agenti di sviluppo per garantire continuità.

---

## 1. Stato Attuale (Cosa è stato implementato)

È stata isolata e configurata un'infrastruttura di test all'interno della cartella `tests/eval` mirata a testare la qualità semantica e logica dell'E-Commerce Agent e del suo engine RAG:

1. **`tests/eval/intent_dataset.json`**
   Creato un database "Ground-Truth" di partenza contenente 5 user-query colloquiali, accompagnate dal relativo dizionario JSON di pretesa estrazione. Funge da banco prova empirico.
   
2. **`tests/eval/test_intent_accuracy.py`**
   Costruito un Python test runner asincrono che, includendo automaticamente l'ambiente `.env`, invia massivamente le query del JSON locale al modulo core `app.services.parser.parse_query_service`. Esegue il rendering automatico delle metriche NLU classificandole in veri e falsi positivi flat, e calcola **F1-Score**, **Precision** e **Recall**.
   
3. **`tests/eval/eval_metrics.py`**
   Sviluppate in via pre-emptive le utilities algoritmiche pure per il comparto matematico del Qdrant Vector Storage. Ospita le equazioni di calcolo pronte per il **MRR (Mean Reciprocal Rank)** e **nDCG@K (Normalized Discounted Cumulative Gain)** per lo smistamento di classifiche prodotto/venditore.

---

## 2. Prossime Fasi (Cosa va fatto - Roadmap QA)

### Fase A: Correzione e Allineamento del Dataset (Priorità Alta)
L'attuale performance del `test_intent_accuracy.py` rasenta la Precision decrescente a causa di una totale disconnessione schematica: `intent_dataset.json` contiene constraint "inventate e piatte" (es. `"is_cheap": true`), mentre il prompt effettivo integrato nel *Parser* restituisce complessi oggetti innestati Pydantic (`"constraints":[{"type":"price", "operator":"<=", "value": 300}]`). 
- **Azione richiesta:** Sostituire l'intero blocco JSON di expectation all'interno del dataset con simulazioni di risposte perfette validate dallo schema Pydantic reale supportato dal parser LLM.

### Fase B: Espansione Data Coverage
Una volta sistemati gli assunti del dataset al punto A, è necessario incrementare verticalmente il volume delle query da far "macinare" all'LLM in fase di valutazione.
- **Azione richiesta:** Ampliare `intent_dataset.json` da 5 query fino a ~30-50 entries, coprendo categorie Edge-Case: query molto brevi, query lunghe con rumore sintattico, e negazioni strette (es. "che non sia usato").

### Fase C: Action Testing del Router ReAct
Attualmente la copertura garantisce che l'Agente "interpreti" bene, manca la certezza che "agisca" coerentemente. Bisogna impedire che si blocchi in cicli infiniti fra chiamate a Tools scorretti.
- **Azione richiesta:** Scrivere unit tests analoghi (es. `test_agent_routing.py`) che somministrino all'Agent Builder una simulazione di dialog history per assicurarsi che operi l'azione decisionale corretta chiamando API in sequenza esatta (e.g. `Seller Analysis Tool` a posto di `Product Search`).

---

## 3. Quali file ispezionare per eseguire le prossime fasi

Prima di intervenire sul debugging dell'agent, dovrai analizzare e riadattare la logica allineando l'output formale del motore principale. Ispeziona e apri questi file:

- 🗂️ `tests/eval/intent_dataset.json`: (Il target principale di riscrittura formale e inserimento dati).
- 🗂️ `tests/eval/test_intent_accuracy.py`: (Per comprendere come l'F1-Score processa i dati e lanciarlo per la convalida dei test aggiornati).
- 🗂️ `app/services/parser.py`: (Cruciale. Contiene la logica interna del parsing LLM, i dict fields permessi e le funzioni che ripuliscono i JSON text).
- 🗂️ `app/agent/schemas.py`: (Cruciale. Definisce rigorosamente i Pydantic base schema per ReAct Agent Response e Intent structure a cui il dataset JSON dovrà obbligatoriamente allinearsi).
