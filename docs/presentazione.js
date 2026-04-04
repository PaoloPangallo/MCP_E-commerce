flowchart TD

subgraph group_backend["Backend"]
node_main["App entry<br/>FastAPI bootstrap<br/>[main.py]"]
node_routes["API routes<br/>HTTP router<br/>[routes.py]"]
node_agent_stream["Agent stream<br/>SSE endpoint<br/>[agent_stream.py]"]
node_auth_router["Auth router<br/>JWT HTTP<br/>[auth_router.py]"]
node_agent_exec["Agent executor<br/>Orchestration service<br/>[agent_executor.py]"]
node_planner["Planner<br/>Task planning<br/>[planner.py]"]
node_memory["Agent memory<br/>Context state<br/>[memory.py]"]
node_tool_registry["Tool registry<br/>Tool binding<br/>[tool_registry.py]"]
  node_mcp_core{ { "MCP core<br/>Tool runtime<br/>[core.py]" } }
node_playwright_server["Playwright server<br/>Browser automation"]
node_search_pipeline["Search pipeline<br/>Ranking pipeline<br/>[search_pipeline.py]"]
node_seller_pipeline["Seller pipeline<br/>Seller intelligence<br/>[seller_pipeline.py]"]
node_ebay_service["eBay service<br/>Marketplace adapter<br/>[ebay.py]"]
  node_rag_store{ { "RAG store<br/>Retrieval stack<br/>[qdrant_store.py]" } }
node_nlp_llm["NLP + LLM<br/>Text understanding<br/>[parser.py]"]
node_models_user[("User model<br/>SQLAlchemy model<br/>[user.py]")]
node_models_listing[("Listing model<br/>SQLAlchemy model<br/>[listing.py]")]
node_db_main[("DB layer<br/>Postgres + Redis<br/>[database.py]")]
end

subgraph group_frontend["Frontend"]
node_ui_app["UI shell<br/>React/Vite app<br/>[App.tsx]"]
node_api_client["API client<br/>HTTP client<br/>[apiClient.ts]"]
node_stream_hook["Stream hook<br/>SSE client<br/>[useAgentStream.ts]"]
node_chat_page["Chat page<br/>Conversation UI<br/>[ChatPage.tsx]"]
node_search_view["Search view<br/>Search UI"]
node_seller_view["Seller view<br/>Seller UI"]
end

subgraph group_runtime["Runtime"]
node_qdrant_data[("Qdrant store<br/>Vector storage")]
end

node_main-- >| "mounts" | node_routes
node_main-- >| "includes" | node_auth_router
node_routes-- >| "streams" | node_agent_stream
node_routes-- >| "search" | node_search_pipeline
node_routes-- >| "seller" | node_seller_pipeline
node_routes-- >| "guards" | node_auth_router
node_agent_stream-- >| "delegates" | node_agent_exec
node_agent_exec-- >| "plans" | node_planner
node_agent_exec-- >| "tracks" | node_memory
node_agent_exec-- >| "uses" | node_tool_registry
node_tool_registry-- >| "invokes" | node_mcp_core
node_mcp_core-- >| "automates" | node_playwright_server
node_mcp_core-- >| "fetches" | node_ebay_service
node_mcp_core-- >| "retrieves" | node_rag_store
node_search_pipeline-- >| "queries" | node_ebay_service
node_search_pipeline-- >| "enriches" | node_rag_store
node_seller_pipeline-- >| "analyzes" | node_ebay_service
node_seller_pipeline-- >| "augments" | node_rag_store
node_nlp_llm-- >| "parses intent" | node_search_pipeline
node_nlp_llm-- >| "supports" | node_agent_exec
node_db_main-- >| "stores" | node_models_user
node_db_main-- >| "stores" | node_models_listing
node_ui_app-- >| "contains" | node_api_client
node_ui_app-- >| "contains" | node_stream_hook
node_stream_hook-- >| "consumes" | node_agent_stream
node_chat_page-- >| "renders" | node_stream_hook
node_search_view-- >| "requests" | node_api_client
node_seller_view-- >| "requests" | node_api_client
node_rag_store-- >| "persists" | node_qdrant_data

click node_main "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/main.py"
click node_routes "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/api/routes.py"
click node_agent_stream "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/api/agent_stream.py"
click node_auth_router "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/auth/auth_router.py"
click node_agent_exec "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/services/agent_executor.py"
click node_planner "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/agent/planner.py"
click node_memory "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/agent/memory.py"
click node_tool_registry "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/agent/tool_registry.py"
click node_mcp_core "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/mcp/core.py"
click node_playwright_server "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/mcp/playwright_server.py"
click node_search_pipeline "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/services/search_pipeline.py"
click node_seller_pipeline "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/services/seller_pipeline.py"
click node_ebay_service "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/services/ebay.py"
click node_rag_store "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/services/rag/qdrant_store.py"
click node_nlp_llm "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/services/parser.py"
click node_models_user "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/models/user.py"
click node_models_listing "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/models/listing.py"
click node_db_main "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/app/db/database.py"
click node_ui_app "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/ebay-ui/src/App.tsx"
click node_api_client "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/ebay-ui/src/api/apiClient.ts"
click node_stream_hook "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/ebay-ui/src/features/agent/hooks/useAgentStream.ts"
click node_chat_page "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/ebay-ui/src/features/chat/ChatPage.tsx"
click node_search_view "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/ebay-ui/src/features/search/components/SearchResultList.tsx"
click node_seller_view "https://github.com/paolopangallo/mcp_e-commerce/blob/Paolo_minimax2.7/ebay-ui/src/features/seller/component/SellerTrustGauge.tsx"
click node_qdrant_data "https://github.com/paolopangallo/mcp_e-commerce/tree/Paolo_minimax2.7/qdrant_storage/collections/ecommerce_rag/0"

classDef toneNeutral fill: #f8fafc, stroke:#334155, stroke - width: 1.5px, color:#0f172a
classDef toneBlue fill: #dbeafe, stroke:#2563eb, stroke - width: 1.5px, color:#172554
classDef toneAmber fill: #fef3c7, stroke: #d97706, stroke - width: 1.5px, color:#78350f
classDef toneMint fill: #dcfce7, stroke:#16a34a, stroke - width: 1.5px, color:#14532d
classDef toneRose fill: #ffe4e6, stroke: #e11d48, stroke - width: 1.5px, color:#881337
classDef toneIndigo fill: #e0e7ff, stroke:#4f46e5, stroke - width: 1.5px, color:#312e81
classDef toneTeal fill: #ccfbf1, stroke:#0f766e, stroke - width: 1.5px, color:#134e4a
class node_main, node_routes, node_agent_stream, node_auth_router, node_agent_exec, node_planner, node_memory, node_tool_registry, node_mcp_core, node_playwright_server, node_search_pipeline, node_seller_pipeline, node_ebay_service, node_rag_store, node_nlp_llm, node_models_user, node_models_listing, node_db_main toneBlue
class node_ui_app, node_api_client, node_stream_hook, node_chat_page, node_search_view, node_seller_view toneAmber
class node_qdrant_data toneMint