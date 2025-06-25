import logging
import os
import json
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from mcp.server.fastmcp import FastMCP
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create MCP server
mcp = FastMCP("postgres-server")

@contextmanager
def get_db_connection():
    """Get database connection with context manager"""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "resturent"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
    try:
        yield conn
    finally:
        conn.close()

@mcp.tool()
async def execute_query(query: str) -> str:
    """Execute a SQL query and return results"""
    logger.info(f"Executing query: {query}")
    try:
        # Basic input validation
        query = query.strip()
        if not query:
            return json.dumps({"error": "Empty query"})
        
        # Prevent dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        if any(keyword in query.upper() for keyword in dangerous_keywords):
            return json.dumps({"error": f"Operation not allowed: {query.split()[0].upper()}"})
        
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                return json.dumps([dict(row) for row in results], indent=2)
            else:
                conn.commit()
                return json.dumps({"message": f"Query executed successfully. Rows affected: {cursor.rowcount}"})
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return json.dumps({"error": f"Error executing query: {str(e)}"})

@mcp.tool()
async def get_table_schema(table_name: str) -> str:
    """Get schema information for a specific table"""
    logger.info(f"Getting schema for table: {table_name}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position;
            """
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
            schema_info = f"Schema for table '{table_name}':\n"
            for col in columns:
                schema_info += f"- {col[0]}: {col[1]} (nullable: {col[2]}, default: {col[3]})\n"
            return schema_info
    except Exception as e:
        logger.error(f"Schema error: {str(e)}")
        return f"Error getting schema: {str(e)}"

@mcp.tool()
async def list_tables() -> str:
    """List all tables in the database"""
    logger.info("Listing tables")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
            cursor.execute(query)
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            return json.dumps({"tables": table_names}, indent=2)
    except Exception as e:
        logger.error(f"List tables error: {str(e)}")
        return f"Error listing tables: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting MCP server...")
    try:
        asyncio.run(mcp.run())
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
    finally:
        logger.info("MCP server stopped.")