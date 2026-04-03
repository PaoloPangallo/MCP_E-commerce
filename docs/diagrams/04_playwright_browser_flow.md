# Sprint 4 — Playwright Browser Contact Flow

```mermaid
sequenceDiagram
    participant User
    participant MCP as "contact_seller_playwright\n(playwright_contact.py)"
    participant PW as "Playwright API"
    participant Chrome as "Chrome Browser\n(real profile)"
    participant eBay as "eBay.it"

    User->>MCP: product_url, message

    alt Chrome has CDP port open (--remote-debugging-port=9222)
        MCP->>Chrome: connect_over_cdp(localhost:9222)
        Chrome-->>MCP: CDP session (already logged in)
        Note over MCP: Uses existing eBay session
    else Launch with real profile
        MCP->>Chrome: launch_persistent_context(user_data_dir)
        Note over Chrome: Requires Chrome fully closed
    end

    MCP->>Chrome: new_page()

    alt IntermediatedFAQ URL
        MCP->>Chrome: page.goto(product_url)
        MCP->>Chrome: click("Non riguarda un oggetto")
        Note over MCP: Tries 9 selector variants
    end

    alt eBay item URL (/itm/)
        MCP->>Chrome: page.goto(product_url)
        MCP->>Chrome: click("Contatta il venditore")
        Note over MCP: Tries 14 selector variants
    end

    alt Login wall detected (signin/login in URL)
        MCP->>User: "Effettua il login, aspetto max 120s"
        User->>Chrome: Manual login
        Chrome-->>MCP: URL no longer contains signin
    end

    MCP->>Chrome: fill(textarea, message)
    Note over MCP: Tries 8 textarea selector variants

    MCP->>Chrome: click("Invia")
    Note over MCP: Tries 7 submit selector variants

    alt Success (submit confirmed)
        MCP->>Chrome: page.close()
        Note over MCP: Only closes the tab, not full browser
        alt not CDP mode
            MCP->>PW: pw.stop()
        end
        MCP-->>User: {success: true, status: "message_sent"}
    else Failure (element not found)
        MCP->>Chrome: leave open for manual intervention
        Note over MCP: Does NOT close browser on failure
        MCP-->>User: {success: false, status, detail}
    end
```

## Connection Strategies

| Strategy | Condition | Pros | Cons |
|----------|-----------|------|------|
| **CDP connect** | Chrome running with `--remote-debugging-port=9222` | No need to close Chrome, preserves session | Chrome must be opened with flag |
| **Persistent context** | Default fallback | Uses real logged-in profile | Chrome must be fully closed |

## Selector Variants Tried

### "Non riguarda un oggetto" button (9 variants)
```
a:has-text('Non riguarda un oggetto')
button:has-text('Non riguarda un oggetto')
a:has-text('Not about an item')
a[href*='notAboutItem']
[data-testid*='not-about-item']
...
```

### Message textarea (8 variants)
```
textarea[name='body']
textarea[id*='message']
textarea[placeholder*='messaggio']
textarea
```

### Submit button (7 variants)
```
button:has-text('Invia')
button:has-text('Send')
input[type='submit']
button[type='submit']
```

## Error States

| Status | Cause |
|--------|-------|
| `login_required` | eBay signin wall detected, timeout after 120s |
| `contact_button_not_found` | "Non riguarda un oggetto" button not found |
| `message_form_not_found` | Textarea not found on page |
| `submit_button_not_found` | Submit button not found |
| `error` | Unexpected exception |
