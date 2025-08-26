from sqlalchemy import text
from app.database import SessionLocal
from app.models import User, Session, Message

def test_database_connection():
    """Test database connection and basic operations"""
    try:
        # Test connection
        db = SessionLocal()
        
        # Test a simple query (fixed for SQLAlchemy 2.0+)
        result = db.execute(text("SELECT 1")).scalar()
        print(f"Database connection successful: {result}")
        
        # Test table existence
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        print(f"Available tables: {tables}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing database connection...")
    if test_database_connection():
        print("✅ Database setup is working correctly!")
    else:
        print("❌ Database setup has issues.")
