Markdown
# MCP E-Commerce · NLP Agent + MCP

Sistema di monitoraggio intelligente prezzi e inserzioni online con parsing NLP, retrieval semantico e orchestrazione agentica.

---

## 🎯 Obiettivo del progetto

Costruire un backend scalabile che:

- **Interpreta** richieste testuali utente.
- **Estrae** vincoli strutturati (prezzo, condizione, prodotto).
- **Recupera** inserzioni da e-commerce (es. eBay).
- **Memorizza** lo storico prezzi.
- **Calcola** trust score venditore/inserzione.
- **Genera** notifiche intelligenti.

---

# ✅ Stato Attuale del Progetto

## 1️⃣ Ambiente Backend

- **Linguaggio:** Python 3.11+
- **Framework:** FastAPI + Uvicorn
- **Struttura:** Architettura modulare (`app/`)
- **Setup:** Virtual environment configurato

**Struttura delle cartelle:**
```text
MCP_ECOM/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   └── parser.py
│   ├── db/
│   │   └── database.py
│   ├── models/
│   └── core/
├── .env
├── requirements.txt
└── README.md
2️⃣ FastAPI Operativo
Endpoint disponibili:

GET /health
Health check del backend.

POST /parse
Parsing NLP della query utente.

Esempio di input:

JSON
{
  "query": "iphone 14 usato sotto 600 euro"
}
Output attuale:

JSON
{
  "original_query": "iphone 14 usato sotto 600 euro",
  "product": "iphone 14",
  "max_price": 600,
  "condition": "usato"
}
3️⃣ NLP Parsing (spaCy)
Modello: it_core_news_sm

Estrazione: Prodotto (NOUN + PROPN), prezzo massimo ("sotto X"), condizione.

Tecnica: Regex + POS tagging.

File: app/services/parser.py

4️⃣ Database & Persistence
PostgreSQL: Database mcp_ecom configurato.

Connessione: Verificata tramite SELECT 1.

Configurazione via .env:

Snippet di codice
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/mcp_ecom
SQLAlchemy: Integrazione completa (Engine, SessionLocal, Base). Creazione automatica delle tabelle inclusa.

File: app/db/database.py

🧠 Architettura Corrente
Snippet di codice
graph TD
    User(User) -->|Text Query| API(FastAPI)
    API -->|Raw Text| NLP(Parser NLP - spaCy)
    NLP -->|Structured Data| API
    API -->|Future| EBAY(eBay Service)
    API -->|Save Data| DB[(PostgreSQL)]
🔜 Roadmap Tecnica (Prossimi Step)
🔵 Fase 1 — Modello Listing
Creare modello SQLAlchemy Listing.

Creazione automatica della tabella.

Endpoint di test per inserimento dati reali.

🔵 Fase 2 — Integrazione eBay API
Registrazione eBay Developer & App ID.

Creazione services/ebay.py.

Endpoint /search con salvataggio su DB.

🔵 Fase 3 — Price History
Tabella price_history.

Tracking variazioni prezzo e trigger notifiche sotto soglia.

🔵 Fase 4 — Retrieval Semantico
Integrazione sentence-transformers.

Generazione embeddings per listing e hybrid search (FAISS).

🔵 Fase 5 — Trust Scoring
Analisi rating venditore e pattern linguistici sospetti.

Generazione score combinato con spiegazione.

🔵 Fase 6 — Agent Orchestrator (MCP)
Implementazione Tool layer e Memory layer.

Ragionamento multi-step ed Explainability.

📦 Stack Tecnologico
Core: FastAPI, SQLAlchemy, PostgreSQL, psycopg3.

NLP: spaCy (it_core_news_sm).

In arrivo: sentence-transformers, FAISS, eBay API, React (frontend), Celery.


Ti serve una mano per scrivere il codice del modello **SQLAlchemy Listing** per la Fase 1?