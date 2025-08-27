import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

load_dotenv()

# Get DATABASE_URL from environment with validation
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Create engine with pool_pre_ping for better connection handling
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Database dependency for FastAPI endpoints
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables (alternative to Alembic for testing)
    """
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
