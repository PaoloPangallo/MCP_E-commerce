# 🧠 Progetto NLP: Monitoraggio Intelligente Prezzi & Inserzioni

## 📋 Overview

**Obiettivo**  
Realizzare un sistema di monitoraggio intelligente di prodotti online che:

- interpreta query in linguaggio naturale
- recupera inserzioni da eBay tramite API ufficiali
- monitora variazioni di prezzo
- applica ranking semantico
- calcola un trust score
- invia notifiche spiegabili

**Approccio tecnico**

- NLP deterministico (spaCy)
- Retrieval ibrido (keyword + embeddings)
- Trust scoring rule-based / ML leggero
- Orchestrazione tool MCP-style (deterministica)
- Valutazione scientifica con metriche IR

---

# 🏗 Architettura Generale


User (React)
↓
FastAPI Backend
↓
spaCy (Parsing Query)
↓
eBay API (Browse)
↓
PostgreSQL (Storage + Price History)
↓
Retrieval Layer (Keyword / Semantic / Hybrid)
↓
Trust Scoring
↓
Event Detection
↓
Notification + Explainability


---

# 🎯 FASE 1 — Setup Dati + Baseline Retrieval

## 1.1 Integrazione eBay API

- Registrazione eBay Developer Program
- Implementazione OAuth client credentials
- Endpoint principale:
  - `/buy/browse/v1/item_summary/search`
- Salvataggio risultati nel database

## 1.2 Database PostgreSQL

### Tabelle principali

```sql
users(
    id SERIAL PRIMARY KEY,
    email TEXT
);

alerts(
    id SERIAL PRIMARY KEY,
    user_id INT,
    query TEXT,
    max_price FLOAT,
    condition TEXT,
    created_at TIMESTAMP
);

listings(
    id TEXT PRIMARY KEY,
    title TEXT,
    price FLOAT,
    seller TEXT,
    seller_score FLOAT,
    condition TEXT,
    timestamp TIMESTAMP
);

price_history(
    id SERIAL PRIMARY KEY,
    listing_id TEXT,
    price FLOAT,
    timestamp TIMESTAMP
);
1.3 Baseline Retrieval

Keyword matching

Filtro prezzo

Ordinamento per score semplice

Metriche iniziali

Precision@5

Recall@10

Deliverable Fase 1

Backend funzionante

Salvataggio inserzioni

Baseline misurabile

🧠 FASE 2 — NLP Query Parsing (spaCy)
2.1 Parsing Linguaggio Naturale

Estrazione:

Prodotto

Prezzo massimo/minimo

Condizione

Piattaforma

Località (se presente)

Esempio output:

{
  "product": "iphone 14",
  "max_price": 600,
  "condition": "usato",
  "platform": "ebay"
}
2.2 Normalizzazione

Mapping sinonimi

Standardizzazione unità

Gestione typo comuni

2.3 Valutazione NLP

Dataset annotato manualmente (50+ query)

Metriche:

Precision extraction

Recall extraction

F1-score

Deliverable Fase 2
Query → Oggetto strutturato valutabile

🔍 FASE 3 — Semantic Retrieval
3.1 Embeddings

sentence-transformers (bge-small / e5-base)

Embedding per titolo e descrizione

3.2 FAISS Index

Creazione indice locale

Similarity search

3.3 Hybrid Search
final_score = α * keyword_score + β * semantic_score
3.4 Metriche IR

Precision@K

Recall@K

MRR

nDCG@K

Confronto:

Versione	Descrizione
Baseline	Keyword only
+ Semantic	Embeddings
Hybrid	Keyword + Semantic

Deliverable Fase 3
Retrieval semantico valutato scientificamente.

🛡 FASE 4 — Trust Scoring + MCP Orchestration
4.1 Trust Scoring
Segnali strutturati

Seller rating

Numero recensioni

Stabilità prezzo

Età inserzione

Formula base
trust_score =
    w1 * seller_rating +
    w2 * log(review_count) +
    w3 * price_stability +
    w4 * text_quality
4.2 Orchestrazione MCP-Style

Pipeline deterministica:

def agent_pipeline(query):
    structured = parse_query(query)
    results = search_listings(structured)
    ranked = hybrid_rank(results)
    trusted = apply_trust(ranked)
    events = detect_event(trusted)
    notify(events)

Tool layer:

search_listings()

get_price_history()

compute_trust()

send_notification()

4.3 Explainability

Ogni notifica include:

Prezzo corrente vs soglia

Storico prezzo

Trust score

Motivo del trigger

📊 Valutazione Finale
User-Based Evaluation

Simulazione 20+ utenti

Precision notifiche

Recall eventi

Feedback qualitativo

Confronto Progressivo
Versione	Precision	Recall
Baseline	X	Y
+ Embeddings	↑	↑
+ Trust	↑	~
+ MCP	stabilizzazione	
🌐 Frontend React

Componenti:

AlertForm

PriceChart

ListingCard

TrustBadge

Funzionalità:

Creazione alert

Dashboard storico prezzi

Visualizzazione trust score

Log notifiche

⚠️ Rischi & Mitigazioni
Rate limiting

Solo API ufficiale eBay

Backoff esponenziale

Caching locale

Dati rumorosi

Deduplica

Normalizzazione testo

Trust senza ground truth

Scoring ibrido

Feedback utente

📅 Timeline Stimata
Fase	Durata
Setup + Baseline	2 settimane
NLP Parsing	1-2 settimane
Semantic Retrieval	2 settimane
Trust + MCP	2 settimane
Totale	7-8 settimane
🎓 Learning Outcomes

Pipeline NLP end-to-end

Retrieval semantico con embeddings

Valutazione IR rigorosa

Orchestrazione MCP-style

Trust scoring explainable

Sistema full-stack integrato

📝 Priorità Strategica

Retrieval + metriche IR

Parsing valutato

Trust scoring semplice

MCP orchestration

UI avanzata
