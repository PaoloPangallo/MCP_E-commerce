import os
import sys
from sqlalchemy import create_engine
from dotenv import load_dotenv

sys.path.append(os.getcwd())

from app.db.database import Base, DATABASE_URL
from app.models.user import User
from app.models.listing import Listing

def init_db():
    print(f"Inizializzando il database: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("Tabelle create con successo!")

if __name__ == "__main__":
    load_dotenv()
    init_db()
