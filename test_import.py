import sys
import os
sys.path.append(os.getcwd())

print("Testing app import...")
try:
    from app.main import app
    print("SUCCESS: App imported successfully.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
