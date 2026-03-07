import requests
import json
import time
import sys

URL = "http://localhost:8040/agent/stream"

QUERY = "iphone massimo 100 euro"

TIMEOUT_CONNECT = 10
TIMEOUT_READ = 60


def test_stream():

    params = {
        "query": QUERY,
        "llm_engine": "ollama"
    }

    print("\n==============================")
    print("TEST SSE STREAM")
    print("==============================\n")

    print("URL:", URL)
    print("Query:", QUERY)
    print()

    start = time.time()

    try:

        print("➡️ Opening connection...")

        response = requests.get(
            URL,
            params=params,
            stream=True,
            timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
            headers={
                "Accept": "text/event-stream"
            }
        )

    except requests.exceptions.ConnectTimeout:
        print("❌ Connection timeout")
        return

    except Exception as e:
        print("❌ Connection error:", e)
        return

    elapsed = time.time() - start

    print("✅ Connection opened")
    print("Status code:", response.status_code)
    print("Time to connect:", round(elapsed, 3), "seconds")

    print("\nResponse headers:\n")

    for k, v in response.headers.items():
        print(f"{k}: {v}")

    print("\n==============================")
    print("STREAM EVENTS")
    print("==============================\n")

    event_count = 0

    try:

        for raw_line in response.iter_lines():

            if not raw_line:
                continue

            line = raw_line.decode("utf-8").strip()

            print("RAW:", line)

            if line.startswith("data:"):

                payload = line[5:].strip()

                try:

                    obj = json.loads(payload)

                    print("\n📦 EVENT", event_count)
                    print(json.dumps(obj, indent=2, ensure_ascii=False))
                    print()

                except Exception:
                    print("⚠️ Not JSON:", payload)

                event_count += 1

    except requests.exceptions.ReadTimeout:

        print("\n⚠️ Read timeout reached")

    except KeyboardInterrupt:

        print("\n⛔ Interrupted by user")

    finally:

        print("\n==============================")
        print("STREAM CLOSED")
        print("Events received:", event_count)
        print("==============================\n")


if __name__ == "__main__":
    test_stream()