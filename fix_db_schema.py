import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL is not set.")
    exit(1)

engine = create_engine(DATABASE_URL)

def run_migration():
    commands = [
        "ALTER TABLE wishlist_items ADD COLUMN IF NOT EXISTS previous_price FLOAT;",
        "ALTER TABLE wishlist_items ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ;",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS contextual_budgets VARCHAR(2000);",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS category_affinities VARCHAR(2000);",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS condition_preference VARCHAR(255);",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS interaction_depth VARCHAR(50);",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS theme VARCHAR(20) DEFAULT 'light' NOT NULL;",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS conversation_tone VARCHAR(50) DEFAULT 'neutral' NOT NULL;"
    ]
    
    with engine.connect() as conn:
        for cmd in commands:
            print(f"Executing: {cmd}")
            try:
                conn.execute(text(cmd))
                conn.commit()
                print("Success.")
            except Exception as e:
                print(f"Failed: {e}")

if __name__ == "__main__":
    run_migration()
