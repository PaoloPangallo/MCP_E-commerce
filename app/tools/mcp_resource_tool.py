import logging
from typing import Any, Dict, Optional
from app.agent.tool_registry import ToolContext

logger = logging.getLogger(__name__)

async def execute_inspect_mcp_resource_tool(action_input: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
    """
    Executor per il tool che legge una risorsa MCP.
    Nota: l'effettiva chiamata viene intercettata dal ToolExecutor se mcp_client è presente.
    Ma qui forniamo una logica di fallback o di interfaccia.
    """
    uri = action_input.get("uri")
    if not uri:
        return {"status": "error", "message": "URI della risorsa mancante."}

    # Il ToolExecutor in app/agent/executor.py gestisce preferibilmente le chiamate MCP.
    # Se arriviamo qui, significa che stiamo cercando di usare il tool 'localmente'
    # ma questo tool ha senso solo se connessi a un server MCP.
    
    return {
        "status": "error",
        "message": "Questo tool richiede un client MCP attivo per leggere la risorsa.",
        "uri": uri
    }
