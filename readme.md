# MCP E-Commerce · AI Shopping Assistant 🛍️

Un assistente intelligente sofisticato (AI Agent) che aiuta gli utenti a trovare le migliori offerte su eBay attraverso l'interpretazione del linguaggio naturale (NLP), il Retrieval Augmented Generation (RAG) e un sistema di classificazione avanzato.

---

## 🚀 Caratteristiche Principali

- **AI Agent (ReAct Framework)**: Un agente autonomo che decide quali strumenti utilizzare (Ricerca, Dettagli, Feedback) per rispondere al meglio alle richieste dell'utente.
- **Parsing NLP Avanzato**: Utilizza **spaCy** e modelli **LLM (Gemini)** per estrarre vincoli strutturati (marca, modello, budget, condizione) da query colloquiali.
- **RAG con Qdrant**: Sistema di recupero semantico basato su **Qdrant** (Vector Database) per analizzare feedback dei venditori e descrizioni prodotti storiche.
- **Scoring & Reranking**: Sistema di pesatura multi-fattore che considera:
  - Rilevanza semantica (Embedding Similarity).
  - Sentiment dei feedback venditore.
  - Seller Trust Score calcolato in tempo reale.
  - Corrispondenza delle preferenze utente (brand preferiti, soglie prezzo).
- **Streaming Response**: Feedback immediato all'utente tramite streaming dei pensieri dell'agente e dei risultati.

---

## 🏗️ Architettura Tecnica

- **Backend**: FastAPI (Python 3.11+)
- **Agent Orchestrator**: Implementazione custom del pattern ReAct con memoria a breve termine.
- **Database Primario**: PostgreSQL (Persistenza inserzioni e profili utente).
- **Vector Database**: Qdrant (Hybrid search: Dense + Sparse BM25).
- **Cache**: Redis (Memorizzazione feedback, sessioni e latenze).
- **NLP**: spaCy (`it_core_news_sm`) + Google Gemini API.

---

## 🛠️ Setup & Installazione

### Requisiti
- Docker Desktop
- Python 3.11+

### Avvio Rapido
1. **Clona il repository**:
   ```bash
   git clone https://github.com/paolo/MCP_ECOM.git
   cd MCP_ECOM
   ```

2. **Configura le variabili d'ambiente**:
   Crea un file `.env` partendo da `.env.example` e inserisci le tue chiavi API (eBay, Gemini, etc.).

3. **Avvia l'ambiente con lo script PowerShell**:
   ```powershell
   ./start_dev.ps1
   ```
   *Questo script avvierà automaticamente i container Docker (Postgres, Redis, Qdrant) e il server FastAPI.*

---

## 📡 API Endpoints (Principali)

| Metodo | Endpoint | Descrizione |
| :--- | :--- | :--- |
| `POST` | `/api/agent/stream` | Endpoint principale per interagire con l'assistente (Streaming). |
| `POST` | `/api/search` | Pipeline di ricerca classica con parsing e persistenza. |
| `GET` | `/api/seller/{name}` | Analisi dettagliata e trust score di un venditore. |
| `GET` | `/health` | Check dello stato dei servizi. |

---

## 🧠 Approccio al Codice & Qualità

Il progetto segue standard elevati di qualità del software:
- **Centralizzazione Config**: TTL della cache e pesi del reranker gestiti in moduli dedicati (`app/config/`).
- **Resilienza**: Fallback automatici e gestione robusta delle eccezioni (es. fallback su database locale se Qdrant è offline).
- **Performance**: Ottimizzazioni sui lookup spaCy e caching intelligente delle query eBay.

---

## 🔜 Sviluppi Futuri
- [ ] Integrazione completa con frontend React (Shopping Dashboard).
- [ ] Supporto multi-marketplace (Amazon, Subito.it).
- [ ] Notifiche push per variazioni prezzo su prodotti salvati.

---
*Creato con ❤️ per semplificare lo shopping online.*