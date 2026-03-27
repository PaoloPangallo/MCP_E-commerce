from app.mcp.tools.playwright_contact import _build_contact_result

def test_build_contact_result_success():
    result = _build_contact_result(
        product_url="https://www.ebay.it/itm/123",
        success=True,
        status="message_sent",
        detail="Messaggio inviato.",
    )
    assert result["status"] == "ok"
    assert result["success"] is True
    assert result["product_url"] == "https://www.ebay.it/itm/123"
    assert result["contact_status"] == "message_sent"

def test_build_contact_result_failure():
    result = _build_contact_result(
        product_url="https://www.ebay.it/itm/123",
        success=False,
        status="login_required",
        detail="eBay richiede il login per contattare i venditori.",
    )
    assert result["status"] == "error"
    assert result["success"] is False
    assert "login" in result["detail"].lower()

def test_build_contact_result_with_message_sent():
    result = _build_contact_result(
        product_url="https://www.ebay.it/itm/123",
        success=True,
        status="message_sent",
        detail="Inviato.",
        message_sent="Ciao, è disponibile?",
    )
    assert result["message_sent"] == "Ciao, è disponibile?"
