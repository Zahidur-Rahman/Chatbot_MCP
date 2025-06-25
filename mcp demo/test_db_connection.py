import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def test_db_connection():
    try:
        # Get connection parameters from environment
        host = os.getenv("POSTGRES_HOST", "localhost")
        database = os.getenv("POSTGRES_DB", "resturent")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        
        print(f"Attempting to connect to PostgreSQL...")
        print(f"Host: {host}")
        print(f"Database: {database}")
        print(f"User: {user}")
        print(f"Port: {port}")
        
        # Try to connect
        if password:
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
        else:
            # Try without password parameter
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                port=port
            )
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        print("✅ Database connection successful!")
        print(f"PostgreSQL version: {version[0]}")
        
        # List tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            print(f"📋 Tables found: {[table[0] for table in tables]}")
        else:
            print("📋 No tables found in the database")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_db_connection() 