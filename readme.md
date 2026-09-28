# MCP E-Commerce — AI Shopping Assistant

**An end-to-end AI shopping system combining MCP, agentic workflows, retrieval and a full web application**

MCP E-Commerce is a working AI-assisted shopping platform built around eBay. It combines a FastAPI backend, a Model Context Protocol server, an agentic orchestration layer, retrieval and ranking services, persistent storage and a React interface.

Unlike a standalone chatbot demo, the project is designed as a complete software system: the AI agent can access explicit tools, search products, inspect sellers, manage user context and interact with application services through a structured architecture.

## What the system does

A user can express a shopping request in natural language and the system can turn it into a sequence of concrete operations.

Depending on the request, the agent can:

- search for products;
- retrieve item details;
- analyse sellers;
- inspect deals and trends;
- work with user profiles and preferences;
- manage wishlists;
- use browser-backed tools where API access is insufficient;
- maintain conversational context;
- stream intermediate and final responses to the frontend.

## Architecture

```text
React / TypeScript UI
        ↓
FastAPI application
        ↓
Agent orchestration
  ┌─────┼──────────────┐
  ↓     ↓              ↓
 MCP   Search / RAG   User context
tools   pipelines      & memory
  ↓        ↓              ↓
eBay   Qdrant        Redis / PostgreSQL
```

The MCP server is mounted directly inside the FastAPI application at `/mcp`, allowing the application to expose shopping capabilities through explicit tools rather than hiding external actions inside prompts.

## MCP tool layer

The project contains dedicated MCP tools for capabilities including:

- product search;
- item inspection;
- seller analysis;
- deals;
- market trends;
- profile information;
- wishlist operations;
- seller contact workflows;
- browser / Playwright-assisted actions;
- conversation context.

This tool-oriented design keeps external actions explicit and makes the agent easier to inspect and extend.

## Agentic layer

The agent package separates responsibilities into dedicated components for:

- planning;
- task decomposition;
- execution;
- tool registration;
- memory;
- prompts and schemas.

This makes the orchestration layer independent from individual integrations and allows new tools or workflows to be introduced without rewriting the entire application.

## Retrieval, ranking and trust

The shopping pipeline uses several signals instead of relying only on keyword matching.

The service layer includes components for:

- semantic retrieval;
- RAG;
- query parsing;
- seller analysis;
- trust scoring;
- sentiment / NLP processing;
- comparison and reranking;
- user profiling;
- price tracking.

**Qdrant** is used for vector retrieval, while **Redis** provides caching and short-lived state and **PostgreSQL** stores persistent application data.

## Frontend

The application includes a modern React interface built with:

- React 19
- TypeScript
- Vite
- Material UI
- Ant Design
- Zustand
- Recharts
- Server-Sent Events / streaming support

The UI is therefore part of the project itself rather than a future integration.

## Backend and infrastructure

- Python
- FastAPI
- Model Context Protocol (MCP)
- PostgreSQL
- Redis
- Qdrant
- SQLAlchemy / Alembic
- eBay integration
- Playwright
- Docker Compose
- Sentence Transformers
- NLP / LLM services

Docker Compose provisions the main infrastructure services locally:

```text
PostgreSQL
Redis
Qdrant
```

The application also performs startup checks, model preloading, shared HTTP-client initialization and background price tracking.

## Repository structure

```text
MCP_E-commerce/
├── app/
│   ├── agent/
│   ├── api/
│   ├── auth/
│   ├── config/
│   ├── db/
│   ├── llm/
│   ├── mcp/
│   ├── models/
│   ├── services/
│   └── tools/
├── ebay-ui/
├── tests/
├── alembic/
└── docker-compose.yaml
```

## Testing

The repository contains dedicated test suites for:

- agent behaviour;
- API behaviour;
- MCP functionality.

## Running locally

The application requires the credentials for the external services used by the project, including the configured eBay and LLM integrations.

Start the infrastructure with:

```bash
docker compose up -d
```

The repository also includes `start_dev.ps1` for the local Windows development workflow.

The React frontend can be started separately:

```bash
cd ebay-ui
npm install
npm run dev
```

## Why this project matters

This is one of the projects that best represents how I like to work: **AI as part of a real software architecture**.

It brings together agentic AI, MCP, retrieval, backend engineering, databases, caching, external APIs, browser automation and frontend development in one functioning end-to-end application.

## Author

**Paolo Pangallo**  
M.Sc. Computer Engineering — Artificial Intelligence  
University of Calabria
