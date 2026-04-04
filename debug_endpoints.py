
import requests
import json

BASE_URL = "http://127.0.0.1:8050"

def test_preferences_endpoint():
    print(f"Testing preferences endpoint at {BASE_URL}/auth/me/preferences...")
    
    try:
        # Check /health first
        resp_h = requests.get(f"{BASE_URL}/health")
        print(f"GET /health: {resp_h.status_code}")
        
        # Test unified Preferences
        resp_pref = requests.patch(f"{BASE_URL}/auth/me/preferences", json={})
        print(f"PATCH /auth/me/preferences: {resp_pref.status_code}")
        print(f"Body: {resp_pref.text}")
        
        # Test individual Instructions (Should be 404 now)
        resp_inst = requests.patch(f"{BASE_URL}/auth/me/instructions", json={})
        print(f"PATCH /auth/me/instructions (Old): {resp_inst.status_code}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_preferences_endpoint()
