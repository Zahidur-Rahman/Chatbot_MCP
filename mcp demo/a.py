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
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from contextlib import asynccontextmanager
import psycopg2
import json

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

# MCP client and process
mcp_client = None
mcp_process = None

class MCPClient:
    """Simple MCP client that communicates with the MCP server process via stdin/stdout"""
    
    def __init__(self, process):
        self.process = process
        self.tools = [
            {"name": "execute_query", "description": "Execute a SQL query and return results"},
            {"name": "get_table_schema", "description": "Get schema information for a specific table"},
            {"name": "list_tables", "description": "List all tables in the database"}
        ]
        self._message_id = 0
    
    async def get_tools(self):
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server"""
        try:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send message to MCP server
            message_str = json.dumps(message) + "\n"
            self.process.stdin.write(message_str)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                if "result" in response:
                    return response["result"]
                elif "error" in response:
                    return {"error": response["error"]}
            
            return {"error": "No response from MCP server"}
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def aclose(self):
        """Close the MCP client"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()

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

async def start_mcp_server():
    """Start the MCP server process"""
    global mcp_process
    try:
        logger.info("Starting MCP server process...")
        
        # Start the MCP server as a separate process
        mcp_process = subprocess.Popen(
            [sys.executable, "postgres_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        logger.info(f"MCP server process started with PID: {mcp_process.pid}")
        
        # Give it a moment to start up
        await asyncio.sleep(2)
        
        # Check if process is still running
        if mcp_process.poll() is not None:
            # Process died, get error output
            stdout, stderr = mcp_process.communicate()
            raise Exception(f"MCP server process died. stdout: {stdout}, stderr: {stderr}")
        
        return mcp_process
        
    except Exception as e:
        logger.error(f"Failed to start MCP server process: {e}")
        if mcp_process:
            try:
                mcp_process.terminate()
            except:
                pass
            mcp_process = None
        return None

async def get_mcp_client():
    """Get or create MCP client"""
    global mcp_client, mcp_process
    
    if mcp_client is None:
        # Try to start MCP server if not running
        if mcp_process is None or mcp_process.poll() is not None:
            mcp_process = await start_mcp_server()
        
        if mcp_process:
            try:
                mcp_client = MCPClient(mcp_process)
                logger.info("MCP client created successfully")
            except Exception as e:
                logger.error(f"Failed to create MCP client: {e}")
                mcp_client = None
    
    return mcp_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        logger.info("Starting application...")
        await get_mcp_client()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise here, allow the app to start without MCP
    yield
    logger.info("Application shutting down")
    # Clean up MCP client and process
    global mcp_client, mcp_process
    if mcp_client:
        try:
            await mcp_client.aclose()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    if mcp_process:
        try:
            mcp_process.terminate()
            mcp_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping MCP process: {e}")
            try:
                mcp_process.kill()
            except:
                pass

app = FastAPI(title="MCP Chatbot API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
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
        
        # Get MCP client and try to use it first
        client = await get_mcp_client()
        tools_used = []
        
        if client:
            # Try to use MCP client
            try:
                # First, let's get the schema information
                schema_info = await get_all_table_schemas()
                schema_text = "\n".join([
                    f"{table}: {', '.join(columns)}" for table, columns in schema_info.items()
                ])
                
                # Generate SQL query using Mistral
                model = get_mistral_model()
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
                
                if sql_query == "Operation not allowed.":
                    return ChatResponse(response="Operation not allowed.", tools_used=[])
                
                # Try to execute via MCP first
                try:
                    result = await client.call_tool("execute_query", {"query": sql_query})
                    tools_used.append("execute_query")
                    
                    if isinstance(result, dict) and 'error' in result:
                        # MCP failed, fall back to direct query
                        db_result = await execute_direct_query(sql_query)
                        tools_used.append("direct_query_fallback")
                    else:
                        db_result = result
                        
                except Exception as mcp_error:
                    logger.warning(f"MCP tool call failed: {mcp_error}")
                    # Fall back to direct query
                    db_result = await execute_direct_query(sql_query)
                    tools_used.append("direct_query_fallback")
                
            except Exception as e:
                logger.error(f"Error with MCP client: {e}")
                # Fall back to direct database access
                return await fallback_chat_processing(request)
        else:
            # No MCP client, use direct database access
            return await fallback_chat_processing(request)
        
        # Format the result
        if isinstance(db_result, dict) and 'error' in db_result:
            return ChatResponse(
                response=f"Generated SQL: {sql_query}\n\nError executing query: {db_result['error']}",
                tools_used=tools_used
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
        elif isinstance(db_result, str):
            # MCP might return JSON string
            try:
                parsed_result = json.loads(db_result)
                if isinstance(parsed_result, list):
                    formatted_result = f"Query Results:\n"
                    formatted_result += f"Rows: {len(parsed_result)}\n"
                    if parsed_result:
                        formatted_result += "Data:\n"
                        for i, row in enumerate(parsed_result[:10]):
                            formatted_result += f"  {i+1}. {row}\n"
                        if len(parsed_result) > 10:
                            formatted_result += f"  ... and {len(parsed_result) - 10} more rows\n"
                else:
                    formatted_result = str(parsed_result)
            except:
                formatted_result = str(db_result)
        else:
            formatted_result = str(db_result)
        
        return ChatResponse(
            response=f"SQL Query: {sql_query}\n\n{formatted_result}",
            tools_used=tools_used
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

async def fallback_chat_processing(request: ChatRequest):
    """Fallback chat processing using direct database access"""
    try:
        schema_info = await get_all_table_schemas()
        schema_text = "\n".join([
            f"{table}: {', '.join(columns)}" for table, columns in schema_info.items()
        ])
        
        model = get_mistral_model()
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
        
        if sql_query == "Operation not allowed.":
            return ChatResponse(response="Operation not allowed.", tools_used=[])
        
        db_result = await execute_direct_query(sql_query)
        
        if isinstance(db_result, dict) and 'error' in db_result:
            return ChatResponse(
                response=f"Generated SQL: {sql_query}\n\nError executing query: {db_result['error']}",
                tools_used=["direct_query"]
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
            tools_used=["direct_query"]
        )
        
    except Exception as e:
        logger.error(f"Fallback chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fallback chat processing error: {str(e)}")

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
                mcp_status = "connected"
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
                logger.info(f"Tables found: {tables}")
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
            "message": "Using direct database access"
        }
        
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        return {
            "tools": [
                {
                    "name": "execute_query",
                    "description": "Execute SQL queries directly against the database"
                }
            ],
            "source": "direct_db",
            "message": f"Error: {str(e)}, using direct database access"
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