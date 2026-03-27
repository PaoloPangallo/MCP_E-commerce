from app.api.agent_stream import StreamRequest, _resolve_mcp_mode


def test_stream_request_default_mcp_mode():
    req = StreamRequest(query="iphone")
    assert req.mcp_mode == "standard"


def test_stream_request_playwright_mode():
    req = StreamRequest(query="iphone", mcp_mode="playwright_browser")
    assert req.mcp_mode == "playwright_browser"


def test_stream_request_invalid_mcp_mode_defaults_to_standard():
    """Unknown mcp_mode values must be sanitized to 'standard'."""
    assert _resolve_mcp_mode("unknown_value") == "standard"
    assert _resolve_mcp_mode("playwright_browser") == "playwright_browser"
    assert _resolve_mcp_mode("standard") == "standard"


def test_stream_request_invalid_mcp_mode_normalised_by_model():
    req = StreamRequest(query="iphone", mcp_mode="unknown_value")
    assert req.mcp_mode == "standard"
