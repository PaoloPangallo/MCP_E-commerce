"""
Model Context Protocol (MCP) Server Entrypoint
This module aggregates the FastMCP engine, its context configuration,
and registers all separate tools and resources under the same MCP instance.
"""

from app.mcp.core import mcp, configure_mcp

# Registry of Tools and Resources
import app.mcp.tools.search
import app.mcp.tools.seller
import app.mcp.tools.item
import app.mcp.tools.metadata
import app.mcp.tools.conversation
import app.mcp.tools.deals
import app.mcp.resources

__all__ = ["mcp", "configure_mcp"]