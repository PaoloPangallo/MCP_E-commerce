# Scaletta Pratica: Progetto NLP Monitoraggio Prezzi & Inserzioni

## 📋 Overview del Progetto

**Obiettivo**: Sistema di monitoraggio intelligente prezzi e inserzioni online con notifiche personalizzate  
**Approccio**: Agent NLP + MCP (Model Context Protocol) + Retrieval semantico  
**Durata stimata**: 4 fasi progressive

---

## 🎯 Fase 1: Setup Dati + Baseline (Fondamenta)

### 1.1 Raccolta Dati
- [ ] Identificare piattaforme target (eBay, Amazon, Subito, ecc.)
- [ ] Implementare connettori API ufficiali (priorità)
- [ ] Setup scraping etico (dove API non disponibile, rate limiting)
- [ ] Creare database PostgreSQL per storage

### 1.2 Storage Prezzi
- [ ] Schema DB: inserzioni + storico prezzi + alert utente
- [ ] Implementare tracking temporale prezzi
- [ ] Sistema di versioning per modifiche inserzioni

### 1.3 Sistema Base
- [ ] Keyword matching semplice (regole + regex)
- [ ] Notifiche base via email/push
- [ ] Dashboard minimale per visualizzazione

**Deliverable Fase 1**: Sistema funzionante con matching keyword + notifiche base

---

## 🧠 Fase 2: NLP Query Parsing (Intelligenza)

### 2.1 Parsing Linguaggio Naturale
- [ ] Implementare spaCy per analisi testo
- [ ] Estrazione entità (NER): prodotti, brand, specifiche
- [ ] Riconoscimento vincoli:
  - Prezzo (max/min/range)
  - Condizione (nuovo/usato/ricondizionato)
  - Piattaforma specifica
  - Località/spedizione

### 2.2 Normalizzazione
- [ ] Mapping sinonimi ("iPhone" = "Apple iPhone")
- [ ] Standardizzazione unità misura
- [ ] Gestione typo comuni

### 2.3 Alternative Avanzate
- [ ] Valutare LLM function calling per parsing complesso
- [ ] Fine-tuning modello HuggingFace se necessario

**Deliverable Fase 2**: Query utente → Parametri strutturati ricerca

---

## 🔍 Fase 3: Semantic Retrieval (Precisione)

### 3.1 Embeddings Semantici
- [ ] Installare sentence-transformers
- [ ] Scegliere modello (bge-small/e5-base)
- [ ] Generare embeddings per descrizioni inserzioni
- [ ] Caching embeddings per ottimizzazione

### 3.2 Vector Search
- [ ] Setup FAISS (locale per prototipo)
- [ ] Configurare Qdrant (se servizio cloud)
- [ ] Indicizzazione vettori
- [ ] Query similarity search

### 3.3 Ranking & Re-ranking
- [ ] Implementare hybrid search (keyword + semantic)
- [ ] Cross-encoder per re-ranking top-K risultati
- [ ] Sperimentare con LLM rerank

### 3.4 Metriche di Valutazione
- [ ] Precision@K e Recall@K
- [ ] MRR (Mean Reciprocal Rank)
- [ ] nDCG per ranking quality
- [ ] A/B test baseline vs semantic

**Deliverable Fase 3**: Retrieval semantico funzionante con metriche IR

---

## 🛡️ Fase 4: Trust Scoring + Agent MCP (Affidabilità)

### 4.1 Trust Scoring Inserzioni
- [ ] Segnali strutturati:
  - Rating venditore
  - Numero transazioni
  - Storico prezzi (anomalie)
  - Età account
- [ ] Analisi testuale:
  - Sentiment RoBERTa
  - Pattern linguistici sospetti
  - Qualità descrizione
- [ ] Combinazione score finale (weighted sum / ML model)

### 4.2 Orchestrazione Agent MCP
- [ ] Definire tool layer MCP:
  - `search_listings()` - ricerca inserzioni
  - `get_price_history()` - storico prezzi
  - `compute_trust()` - calcolo affidabilità
  - `send_notification()` - invio alert
- [ ] Implementare agent orchestrator
- [ ] Logging decisioni per debugging

### 4.3 Spiegabilità Notifiche
- [ ] Report dettagliato per ogni notifica:
  - Perché è stata triggerata
  - Score di trust
  - Confronto prezzi storico
- [ ] Feedback loop utente (utile/non utile)

**Deliverable Fase 4**: Sistema completo con trust scoring e orchestrazione agente

---

## 📊 Valutazione Finale

### User-Based Evaluation
- [ ] Simulare 20+ utenti con preferenze diverse
- [ ] Metriche:
  - **Precision**: notifiche rilevanti / totale notifiche
  - **Recall**: notifiche inviate / eventi rilevanti totali
  - **User satisfaction**: survey qualitativa

### Confronto Progressivo
| Versione | Precision | Recall | Spiegazione |
|----------|-----------|--------|-------------|
| Baseline | ~40% | ~60% | Solo keyword |
| + Embeddings | ~65% | ~75% | Semantic match |
| + Trust | ~80% | ~70% | Filtro qualità |
| + Agent MCP | ~85% | ~80% | Orchestrazione ottimale |

---

## 🚀 Deliverable Finali

1. **Demo Web Interattiva**
   - Dashboard storico prezzi
   - Configurazione alert personalizzati
   - Visualizzazione trust score

2. **Architettura MCP**
   - Tool layer documentato
   - Agent orchestrator con logs
   - API REST per integrazioni

3. **Report Tecnico**
   - Metriche user-based con grafici
   - Confronto baseline → avanzato
   - Limiti e direzioni future
   - Codice open source (GitHub)

---

## 🛠️ Stack Tecnologico Raccomandato

### NLP & ML
- **spaCy** + regex (parsing query)
- **sentence-transformers** (embeddings)
- **FAISS** / Qdrant (vector search)
- **RoBERTa** / LLM (trust scoring)

### Backend & Infra
- **FastAPI** (API server)
- **PostgreSQL** (database)
- **Celery** + APScheduler (monitoring asincrono)
- **Playwright** (scraping se necessario)

### Frontend
- **React** (dashboard)
- **Chart.js** (grafici prezzi)

---

## ⚠️ Rischi e Mitigazioni

### Rischi Principali
1. **Rate limiting / blocchi anti-bot**
   → Usare API ufficiali + rotate IP + caching aggressivo

2. **Dati rumorosi e duplicati**
   → Deduplica con embeddings + fuzzy matching

3. **Drift temporale (descrizioni cambiano)**
   → Re-embeddings periodici + monitoring qualità

4. **Trust scoring senza ground truth**
   → Combinare segnali multipli + feedback loop utente

### Compliance
- Rispettare TOS piattaforme
- GDPR per dati utente
- Trasparenza algoritmi trust

---

## 📅 Timeline Complessiva Stimata

| Fase | Durata | Output |
|------|--------|--------|
| 1 - Setup + Baseline | 2-3 settimane | Sistema funzionante base |
| 2 - NLP Parsing | 1-2 settimane | Query → Parametri strutturati |
| 3 - Semantic Retrieval | 2-3 settimane | Retrieval + metriche IR |
| 4 - Trust + Agent | 2-3 settimane | Sistema completo |
| **Totale** | **7-11 settimane** | **Demo + Report finale** |

---

## 🎓 Learning Outcomes

- Progettazione pipeline NLP end-to-end
- Retrieval semantico (embeddings + vector DB)
- Trust scoring con ML/LLM
- Orchestrazione agenti MCP
- Valutazione user-centric (non solo metriche tecniche)

---

## 📝 Note Finali

Questa scaletta è **modulare**: ogni fase può essere completata indipendentemente. Se hai vincoli di tempo, puoi fermarti a Fase 2-3 per un progetto già significativo. La Fase 4 (agent MCP) è il plus innovativo per distinguere il progetto.

**Priorità**: Qualità > Quantità → Meglio un sistema semplice ma ben valutato che uno complesso ma fragile.
