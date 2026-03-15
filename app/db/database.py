# app/db/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SQL_ECHO = os.getenv("SQL_ECHO", "false").strip().lower() in {"true", "1", "yes"}

engine = create_engine(DATABASE_URL, echo=SQL_ECHO)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()