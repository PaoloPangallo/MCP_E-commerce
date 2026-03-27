import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

print("--- DIAGNOSTIC START ---")
try:
    from app.models.user import User
    print(f"User class imported from: {User.__module__}")
    print(f"File location: {sys.modules['app.models.user'].__file__}")
    
    attrs = [a for a in dir(User) if not a.startswith("__")]
    print(f"User attributes: {attrs}")
    print(f"Has contextual_budgets: {'contextual_budgets' in attrs}")
    
    from app.db.database import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()
    
    # Check table columns
    result = db.execute(text("PRAGMA table_info(users)"))
    columns = [row[1] for row in result.fetchall()]
    print(f"Database 'users' columns: {columns}")
    
    db.close()
except Exception:
    import traceback
    traceback.print_exc()
print("--- DIAGNOSTIC END ---")
