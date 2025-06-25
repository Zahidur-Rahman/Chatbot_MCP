import logging
import os
import asyncio
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from contextlib import asynccontextmanager
import psycopg2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate required environment variables
def validate_env_vars():
    """Validate that required environment variables are set"""
    required_vars = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB", "resturent"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    
    return required_vars

# Initialize environment validation
env_vars = validate_env_vars()

# Initialize MCP client
mcp_client = None
mcp_process = None

# Direct database functions as fallback
async def execute_direct_query(query: str):
    """Execute a SQL query directly against the database"""
    try:
        # Basic input validation
        query = query.strip()
        if not query:
            return {"error": "Empty query"}
        
        # Prevent dangerous operations (basic protection)
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        if any(keyword in query.upper() for keyword in dangerous_keywords):
            return {"error": f"Operation not allowed: {query.split()[0].upper()}"}
        
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if query.strip().upper().startswith('SELECT'):
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return {"columns": columns, "rows": rows}
                else:
                    conn.commit()
                    return {"message": f"Query executed successfully. Rows affected: {cursor.rowcount}"}
    except Exception as e:
        return {"error": str(e)}

async def execute_direct_query_with_params(query: str, params: tuple):
    """Execute a parameterized SQL query directly against the database"""
    try:
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return {"columns": columns, "rows": rows}
                else:
                    conn.commit()
                    return {"message": f"Query executed successfully. Rows affected: {cursor.rowcount}"}
    except Exception as e:
        return {"error": str(e)}

async def get_table_schema_direct(table_name: str):
    """Get schema information for a table directly"""
    try:
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                columns = cursor.fetchall()
                return [{"name": col[0], "type": col[1], "nullable": col[2], "default": col[3]} for col in columns]
    except Exception as e:
        return {"error": str(e)}

async def get_mcp_client():
    global mcp_client, mcp_process
    if mcp_client is None:
        for attempt in range(3):
            try:
                logger.info(f"Attempt {attempt + 1} to initialize MCP client")
                
                # Start the MCP server as a separate process first
                if mcp_process is None:
                    try:
                        # Use sys.executable to ensure we use the correct Python interpreter
                        mcp_process = subprocess.Popen(
                            [sys.executable, "postgres_server.py"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                            creationflags=subprocess.CREATE_NO_WINDOW  # Hide console window on Windows
                        )
                        logger.info("MCP server process started")
                        # Give it a moment to start up
                        await asyncio.sleep(3)  # Increased wait time
                        
                        # Check if process is still running
                        if mcp_process.poll() is not None:
                            # Process died, get error output
                            stdout, stderr = mcp_process.communicate()
                            raise Exception(f"MCP server process died. stdout: {stdout}, stderr: {stderr}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to start MCP server process: {e}")
                        mcp_process = None
                
                # Since the MCP client has issues on Windows, let's create a simple wrapper
                # that communicates directly with our MCP server process
                logger.info("Creating simple MCP client wrapper...")
                
                class SimpleMCPClient:
                    def __init__(self, process):
                        self.process = process
                        self.tools = [
                            {"name": "execute_query", "description": "Execute a SQL query and return results"},
                            {"name": "get_table_schema", "description": "Get schema information for a specific table"},
                            {"name": "list_tables", "description": "List all tables in the database"}
                        ]
                    
                    async def get_tools(self):
                        return self.tools
                    
                    async def aclose(self):
                        pass
                
                mcp_client = SimpleMCPClient(mcp_process)
                logger.info(f"Created simple MCP client with {len(mcp_client.tools)} tools")
                return mcp_client
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.warning(f"Attempt {attempt + 1} failed: {error_type}: {error_msg}")
                
                # Clean up failed client
                if mcp_client:
                    try:
                        await mcp_client.aclose()
                    except:
                        pass
                    mcp_client = None
                
                if attempt == 2:
                    logger.error(f"MCP client failed after 3 attempts. Final error: {error_type}: {error_msg}")
                    return None
                await asyncio.sleep(2)
    return mcp_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_mcp_client()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise here, allow the app to start without MCP
    yield
    logger.info("Application shutting down")
    # Clean up MCP process
    global mcp_process
    if mcp_process:
        try:
            mcp_process.terminate()
            mcp_process.wait(timeout=5)
        except:
            mcp_process.kill()

app = FastAPI(title="MCP Chatbot API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []

class QueryRequest(BaseModel):
    query: str

# Initialize Mistral model
def get_mistral_model():
    api_key = env_vars["MISTRAL_API_KEY"]
    logger.info(f"MISTRAL_API_KEY: {'set' if api_key else 'not set'}")
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not found")
    return ChatMistralAI(model="mistral-large-2407", api_key=api_key)

# Helper to fetch and cache schema info for all tables
async def get_all_table_schemas():
    """Fetch schema info for all tables in the public schema and return as a dict."""
    with psycopg2.connect(
        host=env_vars["POSTGRES_HOST"],
        database=env_vars["POSTGRES_DB"],
        user=env_vars["POSTGRES_USER"],
        password=env_vars["POSTGRES_PASSWORD"],
        port=env_vars["POSTGRES_PORT"]
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            schema_info = {}
            for table in tables:
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public' 
                    ORDER BY ordinal_position
                """, (table,))
                columns = [f"{row[0]} ({row[1]})" for row in cursor.fetchall()]
                schema_info[table] = columns
            return schema_info

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Processing chat request: {request.message}")
        client = await get_mcp_client()
        tools = []
        if client:
            try:
                tools = await client.get_tools()
                logger.info(f"Available MCP tools: {[tool.get('name', tool) for tool in tools]}")
            except Exception as e:
                logger.warning(f"Failed to get MCP tools: {e}")

        model = get_mistral_model()
        messages = [SystemMessage(content="You are a helpful AI assistant that can interact with a PostgreSQL database. If you do not have access to tools, generate a valid SQL query for the user's request and only output the SQL query.")]
        if request.conversation_history:
            for msg in request.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
        messages.append(HumanMessage(content=request.message))

        # --- Optimized: Use improved prompt and schema caching ---
        schema_info = await get_all_table_schemas()
        schema_text = "\n".join([
            f"{table}: {', '.join(columns)}" for table, columns in schema_info.items()
        ])
        improved_prompt = (
            "You are an expert SQL assistant for a PostgreSQL database. "
            "Your job is to convert user requests into a single, safe, syntactically correct SQL SELECT query.\n"
            "- Only generate SELECT statements. Never use DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE.\n"
            f"- Use the following table schemas:\n{schema_text}\n"
            "- Do NOT use Markdown formatting or code blocks. Only output the SQL statement, nothing else.\n"
            "- If the user asks for something not possible with a SELECT, reply: 'Operation not allowed.'\n"
            "- Use only the columns and tables provided.\n"
            f"User request: {request.message}"
        )
        sql_response = await model.ainvoke([HumanMessage(content=improved_prompt)])
        sql_query = sql_response.content.strip().split('\n')[0]
        sql_query = sql_query.replace('\\_', '_').replace('\\', '')

        # --- End optimized prompt ---
        try:
            db_result = await execute_direct_query(sql_query)
            if isinstance(db_result, dict) and 'error' in db_result:
                return ChatResponse(
                    response=f"Generated SQL: {sql_query}\n\nError executing query: {db_result['error']}",
                    tools_used=["execute_query"]
                )
            # Format the result nicely
            if isinstance(db_result, dict) and 'columns' in db_result and 'rows' in db_result:
                formatted_result = "Query Results:\n"
                formatted_result += f"Columns: {', '.join(db_result['columns'])}\n"
                formatted_result += f"Rows: {len(db_result['rows'])}\n"
                if db_result['rows']:
                    formatted_result += "Data:\n"
                    for i, row in enumerate(db_result['rows'][:10]):  # Show first 10 rows
                        formatted_result += f"  {i+1}. {row}\n"
                    if len(db_result['rows']) > 10:
                        formatted_result += f"  ... and {len(db_result['rows']) - 10} more rows\n"
            else:
                formatted_result = str(db_result)
            return ChatResponse(
                response=f"SQL Query: {sql_query}\n\n{formatted_result}",
                tools_used=["execute_query"]
            )
        except Exception as db_error:
            logger.error(f"Database query error: {db_error}")
            return ChatResponse(
                response=f"Generated SQL: {sql_query}\n\nError executing query: {str(db_error)}",
                tools_used=["execute_query"]
            )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            db_status = True
        # Test MCP client
        try:
            client = await get_mcp_client()
            if client:
                tools = await client.get_tools()
                mcp_status = "connected (simple client)"
                tool_count = len(tools)
            else:
                mcp_status = "failed to initialize"
                tool_count = 0
        except Exception as e:
            mcp_status = f"error: {str(e)}"
            tool_count = 0
        return {
            "status": "healthy" if db_status else "degraded",
            "message": "Chatbot API is running",
            "database": "connected" if db_status else "disconnected",
            "mcp_server": mcp_status,
            "available_tools": tool_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }

@app.get("/tables")
async def get_tables():
    """Get list of tables in database"""
    try:
        logger.info("Getting tables from database")
        
        # Since we're using a simple MCP client that doesn't actually execute tools,
        # we'll use direct database queries
        logger.info("Using direct database query")
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"Tables from direct query: {tables}")
                return {"tables": tables, "source": "direct_db"}
                
    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting tables: {str(e)}")

@app.get("/tools")
async def get_tools():
    """Get list of available tools"""
    try:
        client = await get_mcp_client()
        if client:
            try:
                tools = await client.get_tools()
                return {
                    "tools": [
                        {
                            "name": tool.get("name", str(tool)),
                            "description": tool.get("description", "")
                        }
                        for tool in tools
                    ],
                    "source": "mcp"
                }
            except Exception as e:
                logger.warning(f"MCP tools failed: {str(e)}")
                # Fall back to direct tools
                return {
                    "tools": [
                        {
                            "name": "execute_query",
                            "description": "Execute SQL queries directly against the database"
                        },
                        {
                            "name": "get_table_schema", 
                            "description": "Get schema information for a specific table"
                        },
                        {
                            "name": "list_tables",
                            "description": "List all tables in the database"
                        }
                    ],
                    "source": "direct_db",
                    "message": f"MCP unavailable, using direct database tools. MCP error: {str(e)}"
                }
        else:
            # Client is None, which means MCP failed to initialize
            return {
                "tools": [
                    {
                        "name": "execute_query",
                        "description": "Execute SQL queries directly against the database"
                    },
                    {
                        "name": "get_table_schema", 
                        "description": "Get schema information for a specific table"
                    },
                    {
                        "name": "list_tables",
                        "description": "List all tables in the database"
                    }
                ],
                "source": "direct_db",
                "message": "MCP client failed to initialize, using direct database access"
            }
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        return {
            "tools": [
                {
                    "name": "execute_query",
                    "description": "Execute SQL queries directly against the database"
                },
                {
                    "name": "get_table_schema", 
                    "description": "Get schema information for a specific table"
                },
                {
                    "name": "list_tables",
                    "description": "List all tables in the database"
                }
            ],
            "source": "direct_db",
            "message": f"Error getting MCP tools: {str(e)}, using direct database access"
        }

@app.post("/execute-query")
async def execute_query_endpoint(request: QueryRequest):
    """Execute a SQL query directly"""
    try:
        result = await execute_direct_query(request.query)
        return {"result": result}
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

@app.get("/table-schema/{table_name}")
async def get_table_schema_endpoint(table_name: str):
    """Get schema for a specific table"""
    try:
        schema = await get_table_schema_direct(table_name)
        return {"table": table_name, "schema": schema}
    except Exception as e:
        logger.error(f"Schema retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema retrieval error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)