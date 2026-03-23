import requests
import json
import sseclient

def test_stream():
    url = "http://127.0.0.1:8050/agent/stream"
    payload = {
        "query": "iPhone 15",
        "llm_engine": "ollama_cloud",
        "session_id": "test_session",
        "user_id": 1
    }
    
    print(f"Post to {url} with query 'iPhone 15'...")
    response = requests.post(url, json=payload, stream=True)
    client = sseclient.SSEClient(response)
    
    found_results = False
    for event in client.events():
        if event.data:
            try:
                data = json.loads(event.data)
                etype = data.get("type")
                print(f"Event: {etype}")
                
                if etype == "tool_result":
                    tool = data.get("tool")
                    ok = data.get("ok")
                    results = data.get("data", {}).get("results", [])
                    print(f"  Tool: {tool} | ok: {ok} | results_count: {len(results)}")
                    if tool == "search_products" and len(results) > 0:
                        found_results = True
                
                if etype == "final":
                    final_results = data.get("final_data", {}).get("search", {}).get("results", [])
                    print(f"  Final Results Count: {len(final_results)}")
                    if len(final_results) > 0:
                        found_results = True
                    break
            except Exception as e:
                print(f"  Error parsing event: {e}")

    if found_results:
        print("\nSUCCESS: Results found in SSE stream!")
    else:
        print("\nFAILURE: No results found in SSE stream.")

if __name__ == "__main__":
    test_stream()
