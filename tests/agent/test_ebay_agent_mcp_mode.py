# tests/agent/test_ebay_agent_mcp_mode.py
import pytest
from unittest.mock import MagicMock


def test_ebay_agent_stores_mcp_mode():
    from app.agent.ebay_agent import EbayReactAgent
    db = MagicMock()
    agent = EbayReactAgent(db=db, mcp_mode="playwright_browser")
    assert agent.mcp_mode == "playwright_browser"


def test_ebay_agent_default_mcp_mode_is_standard():
    from app.agent.ebay_agent import EbayReactAgent
    db = MagicMock()
    agent = EbayReactAgent(db=db)
    assert agent.mcp_mode == "standard"


def test_ebay_agent_playwright_mode_uses_playwright_url():
    from app.agent.ebay_agent import EbayReactAgent
    db = MagicMock()
    agent = EbayReactAgent(db=db, mcp_mode="playwright_browser")
    assert "/playwright/mcp" in agent.mcp_server_url
