# Backend Activity Diagram — MCP_ECOM

> Mermaid activity diagrams. Render in GitHub, GitLab, or any Mermaid viewer.
> Or paste into [mermaid.live](https://mermaid.live) for interactive preview.

---

## 1. Agent Stream — Richiesta `/agent/stream`

```mermaid
activity-beta
start
:HTTP POST /agent/stream;

note right
  EbayReactAgent.run_stream()
  runs as asyncio background task
end note

:Acquire semaphore (concurrency limit);

if (semaphore available?) then (no)
  :Return 429 Too Many Requests;
  stop
else (yes)
endif

fork
  :Background: heartbeat every 5s;
fork again
  :Main: run ReAct loop;
end fork

repeat :ReAct Step (max 6 iterations)
  if (image attached?) then (yes)
    :Vision Pre-processing;
    :Qwen-VL image analysis;
    :Enrich query with vision context;
  endif

  :Memory hydration;
  :Load session + long-term memory from Redis;

  :Task decomposition;
  :deterministic multi-step planning;

  :ReactPlanner.decide();
  :LLM selects next action via MCP tool catalog;

  if (action == "stop") then (yes)
    :_finalize_stream();
    :LLM synthesis of all observations;
    :Yield SSE chunks to client;
    stop
  else (no - tool call)
    :ToolExecutor.execute_many();
    :MCPToolClient.call_tool_async();
    if (local tool?) then (yes)
      :FastMCP direct call;
    else (HTTP)
      :streamable_http_client → MCP Server;
    endif
    :Service execution;
    :Redis cache check/store;

    :Memory.apply_observation;
    :Update session state in Redis;

    if (terminal condition met?) then (yes)
      :_finalize_stream;
      stop
    else (no)
    endif
  endif
repeat while (max steps not reached)

:Hard timeout at 380s;
stop
```

---

## 2. Search Pipeline — Ricerca Prodotti

```mermaid
activity-beta
start
:POST /search — run_search_pipeline();

:parse_query_service();
:LLM semantic parsing (Redis cached 1h);

if (cache hit?) then (yes)
  :Return cached result;
  stop
else (no)
endif

:ebay.search_items();
:Browse API call;

:Parallel RAG reranking;
fork
  :Vector embedding + similarity;
fork again
  :BM25 keyword matching;
end fork

:Hybrid fusion (reciprocal rank);

:Seller trust scoring;
:feedback ratio + recency + confidence;

:Normalize output;
:Cache tool_result in Redis;
:Return SearchResponse;
stop
```

---

## 3. Seller Analysis Pipeline

```mermaid
activity-beta
start
:analyze_seller(seller_id);

parallel :Parallel fetch (all at once)
  :ebay.get_seller_feedback();
  :ebay.get_user_details();
  :ebay.get_store_details();
end parallel

:Sentiment analysis;
:NLP on feedback text;

:Trust score computation;
:feedback_ratio * recency * confidence;

:Normalize scores;
:Return SellerAnalysis;
stop
```

---

## 4. MCP Tool Call Flow

```mermaid
activity-beta
start
:MCPToolClient.call_tool_async(tool_name, args);

:Build cache key;
if (Redis cache hit?) then (yes)
  :Return cached result;
  stop
else (no)
endif

if (is_local_mode?) then (yes)
  :_get_local_mcp_instance();
  :FastMCP direct tool call;
else (HTTP mode)
  :streamable_http_client;
  :ClientSession.call_tool();
endif

:Service execution;
:_extract_content_parts;
:_parse_parts_to_dict;

:Cache result in Redis (TTL from config);

:Return ToolResult;
stop
```

---

## 5. Compare Pipeline — Confronto Prodotti

```mermaid
activity-beta
start
:compare_products(query_list);

parallel :For each query in parallel
  :run_search_pipeline(query);
end parallel

:Rerank all results together;
:Reciprocal rank fusion;

:Winner selection;
:Score-based product comparison;

:Return CompareResponse;
stop
```

---

## 6. User Profiling & Memory

```mermaid
activity-beta
start
:User query arrives;

:Update session_memory in Redis;
:Store recent query, sellers, products;

if (preference signal detected?) then (yes)
  :Auto-learn preference;
  :Update long_term_memory;
  :category_affinities, budget, condition;
endif

:Return enriched context to agent;
stop
```

---

## 7. Dual MCP World Architecture

```mermaid
activity-beta
start
:Frontend selects mcp_mode;

if (mcp_mode == "standard") then (yes)
  :Route to /mcp/standard;
  :FastMCP (mcp) instance;
  :All tools available;
else if (mcp_mode == "playwright_browser") then (yes)
  :Route to /mcp/playwright;
  :FastMCP (mcp_playwright) instance;
  :search_products (visible browser);
  :contact_seller_playwright;
  :Other tools synced from standard;
endif

:MCPToolClient.call_tool_async;
stop
```

---

## 8. Startup — Application Lifespan

```mermaid
activity-beta
start
:uvicorn starts app;

:Preload SentenceTransformer;
:Create DB tables (SQLAlchemy);
:Check Redis connection;
:Init eBay HTTP client;

fork
  :Background: price tracking loop;
fork again
  :FastAPI ready on port 8050;
end fork

:Waiting for requests...;
stop
```
