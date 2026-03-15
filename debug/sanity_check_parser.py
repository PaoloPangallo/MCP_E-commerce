import os
import inspect
from pathlib import Path

print("\n==============================")
print(" SANITY CHECK PARSER")
print("==============================\n")

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------
# 1) FIND ALL parse_query_service CALLS
# --------------------------------------------------

print("🔎 Searching for parse_query_service calls...\n")

for path in ROOT.rglob("*.py"):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "parse_query_service(" in line:
            print(f"{path}:{i+1}")
            print("   ", line.strip())

print("\n")

# --------------------------------------------------
# 2) FIND llm_engine REFERENCES
# --------------------------------------------------

print("🔎 Searching for llm_engine usage...\n")

for path in ROOT.rglob("*.py"):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "llm_engine" in line:
            print(f"{path}:{i+1}")
            print("   ", line.strip())

print("\n")

# --------------------------------------------------
# 3) CHECK PARSER SIGNATURE
# --------------------------------------------------

print("🔎 Checking parser signature...\n")

try:
    from app.services.parser import parse_query_service

    sig = inspect.signature(parse_query_service)
    print("Parser signature:")
    print(sig)

except Exception as e:
    print("❌ Failed importing parser:", e)

print("\n")

# --------------------------------------------------
# 4) TEST CALL
# --------------------------------------------------

print("🔎 Testing parser call...\n")

try:
    result = parse_query_service(
        "iphone 13 massimo 1000 euro",
        use_llm=False,
        include_meta=True
    )

    print("✅ Parser works")
    print(result)

except Exception as e:
    print("❌ Parser call failed:", e)

print("\n==============================")
print(" END SANITY CHECK")
print("==============================")