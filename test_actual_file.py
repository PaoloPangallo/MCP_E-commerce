
with open('app/agent/planner.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("--- Line 19 (VALID_INTENTS) ---")
print(lines[18].strip())

print("\n--- Lines 337-375 (_deterministic_decide) ---")
for i in range(336, 375):
    print(f"{i+1}: {lines[i].strip()}")

print("\n--- Lines 163-180 (top_two) ---")
for i in range(162, 180):
    print(f"{i+1}: {lines[i].strip()}")
