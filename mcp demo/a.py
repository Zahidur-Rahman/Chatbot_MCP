import logging
import os
import asyncio
import subprocess
import sys
import time
import platform
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from contextlib import asynccontextmanager
import json
import threading
import queue
import datetime
import asyncpg
import signal
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Cross-platform event loop policy setup
def setup_event_loop():
    """Setup appropriate event loop policy based on platform"""
    if platform.system() == "Windows":
        # Use ProactorEventLoop on Windows for better subprocess handling
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.info("✅ Windows event loop policy set")
    else:
        # Use default policy on Unix-like systems
        logger.info("✅ Using default event loop policy for Unix-like system")

# Setup event loop early
setup_event_loop()

# Environment validation with better error handling
@lru_cache(maxsize=1)
def validate_env_vars():
    """Validate that required environment variables are set"""
    required_vars = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB", "restaurant"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        if "MISTRAL_API_KEY" in missing_vars:
            raise ValueError("MISTRAL_API_KEY is required")
    
    # Validate database port
    try:
        required_vars["POSTGRES_PORT"] = int(required_vars["POSTGRES_PORT"])
    except ValueError:
        logger.error("❌ POSTGRES_PORT must be a valid integer")
        raise ValueError("POSTGRES_PORT must be a valid integer")
    
    return required_vars

# Initialize environment validation
env_vars = validate_env_vars()

# Global variables
mcp_client = None
mcp_process = None
db_pool = None
schema_cache = {
    "tables": {},
    "schemas": {},
    "table_suggestions": {},
    "last_updated": None
}

class SimpleMCPClient:
    """Cross-platform MCP client with improved error handling"""
    
    def __init__(self, process):
        self.process = process
        self.tools = [
            {"name": "execute_query", "description": "Execute a SQL query and return results"},
            {"name": "get_table_schema", "description": "Get schema information for a specific table"},
            {"name": "list_tables", "description": "List all tables in the database"}
        ]
        self._message_id = 0
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._communication_thread = None
        self._initialized = False
        self._lock = threading.Lock()
        self._start_communication_thread()
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize the MCP server with proper handshake"""
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "fastapi-client", "version": "1.0.0"}
                }
            }
            
            init_str = json.dumps(init_message) + "\n"
            self.process.stdin.write(init_str)
            self.process.stdin.flush()
            
            # Wait for initialization response with timeout
            start_time = time.time()
            timeout = 10
            
            while time.time() - start_time < timeout:
                if self.process.stdout.readable():
                    response_line = self.process.stdout.readline()
                    if response_line:
                        try:
                            response = json.loads(response_line.strip())
                            if response.get("id") == 1 and "result" in response:
                                logger.info("✅ MCP server initialized successfully")
                                self._initialized = True
                                
                                # Send initialized notification
                                initialized_notification = {
                                    "jsonrpc": "2.0",
                                    "method": "notifications/initialized"
                                }
                                notif_str = json.dumps(initialized_notification) + "\n"
                                self.process.stdin.write(notif_str)
                                self.process.stdin.flush()
                                return
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse initialization response: {e}")
                            continue
                time.sleep(0.1)
            
            logger.error("❌ MCP server initialization timeout")
                
        except Exception as e:
            logger.error(f"❌ MCP initialization error: {e}")
    
    def _start_communication_thread(self):
        """Start a background thread to handle MCP communication"""
        def communication_worker():
            try:
                while True:
                    try:
                        request = self._request_queue.get(timeout=1)
                        if request is None:  # Poison pill to stop thread
                            break
                        
                        with self._lock:
                            if self.process.poll() is not None:
                                logger.error("❌ MCP process died")
                                self._response_queue.put({"error": "MCP process died"})
                                break
                            
                            # Send request to MCP server
                            request_str = json.dumps(request) + "\n"
                            self.process.stdin.write(request_str)
                            self.process.stdin.flush()
                            
                            # Read response with timeout
                            start_time = time.time()
                            timeout = 15
                            
                            while time.time() - start_time < timeout:
                                if self.process.stdout.readable():
                                    response_line = self.process.stdout.readline()
                                    if response_line:
                                        try:
                                            response = json.loads(response_line.strip())
                                            self._response_queue.put(response)
                                            break
                                        except json.JSONDecodeError:
                                            continue
                                time.sleep(0.1)
                            else:
                                self._response_queue.put({"error": "Response timeout"})
                            
                    except queue.Empty:
                        # Check if process is still alive
                        if self.process.poll() is not None:
                            logger.error("❌ MCP process died")
                            break
                        continue
                    except Exception as e:
                        logger.error(f"❌ Communication thread error: {e}")
                        self._response_queue.put({"error": str(e)})
                        
            except Exception as e:
                logger.error(f"❌ Communication worker fatal error: {e}")
        
        self._communication_thread = threading.Thread(target=communication_worker, daemon=True)
        self._communication_thread.start()
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server with improved error handling"""
        if not self._initialized:
            return {"error": "MCP server not initialized"}
        
        try:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments}
            }
            
            self._request_queue.put(message)
            
            # Wait for response with timeout
            timeout = 30
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=1)
                    if response.get("id") == self._message_id:
                        if "result" in response:
                            return response["result"]
                        elif "error" in response:
                            return {"error": response["error"]}
                except queue.Empty:
                    continue
            
            return {"error": "Request timeout"}
            
        except Exception as e:
            logger.error(f"❌ Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def aclose(self):
        """Close the MCP client gracefully"""
        if self._communication_thread and self._communication_thread.is_alive():
            self._request_queue.put(None)  # Poison pill
            self._communication_thread.join(timeout=5)
        
        if self.process and self.process.poll() is None:
            try:
                # Cross-platform process termination
                if platform.system() == "Windows":
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGTERM)
                
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.error(f"Error closing MCP process: {e}")

# Database functions with connection pooling
async def create_db_pool():
    """Create database connection pool"""
    try:
        pool = await asyncpg.create_pool(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"],
            min_size=2,
            max_size=10,
            command_timeout=30
        )
        logger.info("✅ Database connection pool created")
        return pool
    except Exception as e:
        logger.error(f"❌ Failed to create database pool: {e}")
        raise

async def test_database_connection():
    """Test database connection and log status"""
    global db_pool
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def execute_direct_query(query: str):
    """Execute a SQL query directly against the database with better error handling"""
    global db_pool
    try:
        logger.info(f"🔍 Executing query: {query[:100]}...")
        
        if not query.strip():
            return {"error": "Empty query"}
        
        if not query.strip().upper().startswith('SELECT'):
            return {"error": "Only SELECT queries allowed"}

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query)
            columns = list(rows[0].keys()) if rows else []
            result_rows = [dict(row) for row in rows]
            
            return {
                "columns": columns,
                "rows": result_rows,
                "row_count": len(result_rows)
            }
    except Exception as e:
        logger.error(f"❌ Database query error: {e}")
        return {"error": str(e)}

# Enhanced schema management
async def build_enhanced_schema_cache():
    """Build comprehensive schema cache with fuzzy matching support"""
    global schema_cache, db_pool
    
    try:
        async with db_pool.acquire() as conn:
            # Get all tables
            tables = await conn.fetch("""
                SELECT table_name, table_type
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            schema_info = {}
            table_suggestions = {}
            
            for table in tables:
                table_name = table['table_name']
                
                # Get column information
                columns = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = $1 AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, table_name)
                
                column_info = []
                for col in columns:
                    column_info.append({
                        "name": col['column_name'],
                        "type": col['data_type'],
                        "nullable": col['is_nullable'] == 'YES',
                        "default": col['column_default']
                    })
                
                schema_info[table_name] = {
                    "columns": column_info,
                    "type": table['table_type']
                }
                
                # Build fuzzy matching suggestions
                table_keywords = []
                table_keywords.append(table_name.lower())
                table_keywords.extend([word.lower() for word in table_name.split('_')])
                table_keywords.extend([col['name'].lower() for col in column_info])
                
                table_suggestions[table_name] = {
                    "keywords": list(set(table_keywords)),
                    "display_name": table_name,
                    "description": f"Table with {len(column_info)} columns"
                }
        
        schema_cache = {
            "tables": {table['table_name']: table for table in tables},
            "schemas": schema_info,
            "table_suggestions": table_suggestions,
            "last_updated": datetime.datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Enhanced schema cache built with {len(tables)} tables")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building schema cache: {e}")
        return False

def find_matching_tables(user_input: str) -> List[Dict[str, Any]]:
    """Find tables that match user input using fuzzy matching"""
    user_words = set(word.lower() for word in user_input.split() if len(word) > 2)
    matches = []
    
    for table_name, suggestion in schema_cache["table_suggestions"].items():
        score = 0
        matched_keywords = []
        
        for keyword in suggestion["keywords"]:
            for user_word in user_words:
                if user_word in keyword or keyword in user_word:
                    score += 1
                    matched_keywords.append(keyword)
                    break
        
        if score > 0:
            matches.append({
                "table_name": table_name,
                "score": score,
                "matched_keywords": matched_keywords,
                "suggestion": suggestion
            })
    
    # Sort by score (descending)
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:5]  # Return top 5 matches

# Enhanced prompt generation
def generate_enhanced_sql_prompt(user_message: str, conversation_history: List = None) -> str:
    """Generate an enhanced SQL prompt with better context and fuzzy matching"""
    
    # Find matching tables
    matching_tables = find_matching_tables(user_message)
    
    # Build schema context
    schema_text = "Available tables and their schemas:\n\n"
    
    if matching_tables:
        # Prioritize matching tables
        schema_text += "📍 MOST RELEVANT TABLES FOR YOUR QUERY:\n"
        for match in matching_tables:
            table_name = match["table_name"]
            if table_name in schema_cache["schemas"]:
                columns = schema_cache["schemas"][table_name]["columns"]
                column_desc = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
                schema_text += f"• {table_name}: {column_desc}\n"
        schema_text += "\n"
    
    # Add all tables for reference
    schema_text += "📋 ALL AVAILABLE TABLES:\n"
    for table_name, schema_info in schema_cache["schemas"].items():
        columns = schema_info["columns"]
        column_desc = ", ".join([f"{col['name']} ({col['type']})" for col in columns[:5]])  # First 5 columns
        if len(columns) > 5:
            column_desc += f" ... and {len(columns) - 5} more"
        schema_text += f"• {table_name}: {column_desc}\n"
    
    # Enhanced prompt with examples
    prompt = f"""You are an expert SQL assistant for a PostgreSQL database. Your job is to convert user requests into safe, syntactically correct SQL SELECT queries.

IMPORTANT RULES:
- Generate ONLY SELECT statements. Never use DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE
- Use ONLY the tables and columns provided in the schema below
- If unsure about table/column names, use the most relevant match from the available options
- When user mentions partial names, match them to the closest table/column names available
- Always use proper PostgreSQL syntax with double quotes for identifiers if needed
- Include reasonable LIMIT clauses for large result sets (default: LIMIT 100)

{schema_text}

QUERY EXAMPLES:
- For "show customers": SELECT * FROM customers LIMIT 100;
- For "find orders": SELECT * FROM orders LIMIT 100;
- For "menu items": SELECT * FROM menu_items LIMIT 100;
- For "customer named John": SELECT * FROM customers WHERE name ILIKE '%John%';

USER REQUEST: {user_message}

RESPONSE FORMAT:
- If the request can be converted to a SELECT query, return ONLY the SQL statement
- If the request cannot be safely converted, respond with exactly: "Operation not allowed"
- Do NOT use markdown formatting or code blocks
- Do NOT explain the query unless specifically asked

SQL Query:"""
    
    return prompt

# Cross-platform MCP server startup
async def start_mcp_server():
    """Start the MCP server process with cross-platform compatibility"""
    global mcp_process
    
    try:
        logger.info("🚀 Starting MCP server process...")
        
        script_name = "postgres_server.py"
        if not os.path.exists(script_name):
            logger.warning(f"⚠️ MCP server file {script_name} not found")
            return None
        
        # Cross-platform subprocess configuration
        if platform.system() == "Windows":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            close_fds = False
        else:
            creation_flags = 0
            close_fds = True
        
        mcp_process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=0,
            universal_newlines=True,
            creationflags=creation_flags,
            close_fds=close_fds
        )
        
        logger.info(f"✅ MCP server process started with PID: {mcp_process.pid}")
        
        # Wait for startup
        await asyncio.sleep(2)
        
        if mcp_process.poll() is not None:
            stdout, stderr = mcp_process.communicate()
            logger.error(f"❌ MCP server process died. stdout: {stdout}, stderr: {stderr}")
            raise Exception(f"MCP server process died: {stderr}")
        
        logger.info("✅ MCP server is running successfully")
        return mcp_process
        
    except Exception as e:
        logger.error(f"❌ Failed to start MCP server: {e}")
        if mcp_process:
            try:
                mcp_process.terminate()
            except:
                pass
            mcp_process = None
        return None

async def get_mcp_client():
    """Get or create MCP client with better error handling"""
    global mcp_client, mcp_process
    
    if mcp_client is None or (hasattr(mcp_client, 'process') and mcp_client.process.poll() is not None):
        # Start or restart MCP server
        if mcp_process is None or mcp_process.poll() is not None:
            mcp_process = await start_mcp_server()
        
        if mcp_process:
            try:
                mcp_client = SimpleMCPClient(mcp_process)
                await asyncio.sleep(1)  # Give client time to initialize
                logger.info("✅ MCP client created successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create MCP client: {e}")
                mcp_client = None
    
    return mcp_client

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with enhanced startup/shutdown"""
    global db_pool
    
    try:
        logger.info("🚀 Starting application...")
        
        # Create database pool
        db_pool = await create_db_pool()
        
        # Test database connection
        db_connected = await test_database_connection()
        if not db_connected:
            logger.error("❌ Database connection failed on startup")
            raise Exception("Database connection failed")
        
        # Build enhanced schema cache
        schema_built = await build_enhanced_schema_cache()
        if not schema_built:
            logger.warning("⚠️ Schema cache build failed")
        
        # Try to initialize MCP
        client = await get_mcp_client()
        if client:
            logger.info("✅ MCP tools initialized successfully")
        else:
            logger.warning("⚠️ MCP tools not available, using direct database access")
        
        logger.info("✅ Application startup completed")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Application shutting down...")
    
    if db_pool:
        await db_pool.close()
        logger.info("🛑 Database pool closed")
    
    global mcp_client, mcp_process
    if mcp_client:
        try:
            await mcp_client.aclose()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    
    if mcp_process:
        try:
            if platform.system() == "Windows":
                mcp_process.terminate()
            else:
                mcp_process.send_signal(signal.SIGTERM)
            mcp_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping MCP process: {e}")
            try:
                mcp_process.kill()
            except:
                pass

# FastAPI app initialization
app = FastAPI(
    title="Enhanced MCP Chatbot API",
    version="2.0.0",
    description="Cross-platform chatbot API with enhanced SQL query generation",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    suggested_tables: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str

# Mistral model initialization
@lru_cache(maxsize=1)
def get_mistral_model():
    """Get Mistral model instance with caching"""
    api_key = env_vars["MISTRAL_API_KEY"]
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not found")
    return ChatMistralAI(
        model="mistral-large-2407",
        api_key=api_key,
        temperature=0.1,  # Lower temperature for more consistent SQL generation
        max_tokens=1000
    )

# Enhanced chat processing
async def enhanced_chat_processing(request: ChatRequest):
    """Enhanced chat processing with better SQL query generation"""
    try:
        # Generate enhanced prompt
        enhanced_prompt = generate_enhanced_sql_prompt(
            request.message,
            request.conversation_history
        )
        
        model = get_mistral_model()
        sql_response = await model.ainvoke([HumanMessage(content=enhanced_prompt)])
        sql_query = sql_response.content.strip()
        
        # Clean up the SQL query
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        if sql_query == "Operation not allowed":
            return await general_chat_processing(request)
        
        # Find matching tables for response metadata
        matching_tables = find_matching_tables(request.message)
        suggested_table_names = [match["table_name"] for match in matching_tables]
        
        logger.info(f"🔍 Generated SQL: {sql_query}")
        
        # Try MCP first, then fallback to direct query
        client = await get_mcp_client()
        if client:
            query_result = await client.call_tool("execute_query", {"query": sql_query})
            result = await process_mcp_result(query_result)
        else:
            result = await execute_direct_query(sql_query)
        
        if isinstance(result, dict) and result.get("error"):
            # Try to suggest corrections
            error_msg = result["error"]
            if "does not exist" in error_msg.lower():
                suggestion_text = ""
                if matching_tables:
                    suggestion_text = f" Did you mean one of these tables: {', '.join(suggested_table_names[:3])}?"
                return ChatResponse(
                    response=f"Table or column not found.{suggestion_text} Please check the available tables and try again.",
                    tools_used=["error_with_suggestion"],
                    suggested_tables=suggested_table_names
                )
            else:
                return ChatResponse(
                    response="I encountered an error processing your query. Please try rephrasing your request.",
                    tools_used=["error"],
                    suggested_tables=suggested_table_names
                )
        
        # Format successful result
        if result.get("rows"):
            formatted_result = format_query_results(result["rows"], limit=10)
            row_count = result.get("row_count", len(result["rows"]))
            
            if row_count > 10:
                formatted_result += f"\n\n... and {row_count - 10} more rows (showing first 10)"
            
            return ChatResponse(
                response=formatted_result,
                tools_used=["enhanced_sql_query"],
                suggested_tables=suggested_table_names
            )
        else:
            return ChatResponse(
                response="No results found for your query.",
                tools_used=["enhanced_sql_query"],
                suggested_tables=suggested_table_names
            )
        
    except Exception as e:
        logger.error(f"❌ Enhanced chat processing error: {e}")
        return await general_chat_processing(request)

async def process_mcp_result(query_result):
    """Process MCP query result with better error handling"""
    if isinstance(query_result, dict):
        if 'error' in query_result:
            return {"error": query_result['error']}
        
        if 'content' in query_result:
            content = query_result['content']
            if isinstance(content, list) and len(content) > 0:
                try:
                    result_data = json.loads(content[0]['text']) if content[0].get('text') else {}
                except json.JSONDecodeError:
                    return {"error": "Invalid response format"}
            else:
                result_data = {}
        else:
            result_data = query_result
        
        if isinstance(result_data, dict):
            if 'error' in result_data:
                return {"error": result_data['error']}
            elif 'results' in result_data:
                return {
                    "rows": result_data['results'],
                    "row_count": len(result_data['results'])
                }
    
    return {"error": "Unexpected response format"}

def format_query_results(rows: List[Dict], limit: int = 10) -> str:
    """Format query results for display"""
    if not rows:
        return "No results found."
    
    # Limit results
    display_rows = rows[:limit]
    
    # Format as a simple table
    if len(display_rows) == 1:
        # Single row - format as key-value pairs
        result = ""
        for key, value in display_rows[0].items():
            result += f"{key}: {value}\n"
        return result.strip()
    else:
        # Multiple rows - format as a simple list
        result = ""
        for i, row in enumerate(display_rows, 1):
            result += f"{i}. {dict(row)}\n"
        return result.strip()

async def general_chat_processing(request: ChatRequest):
    """Handle general non-database chat with context awareness"""
    try:
        model = get_mistral_model()
        
        # Build conversation context
        messages = [
            SystemMessage(content="You are a helpful assistant. If the user asks about database or data-related queries, suggest they be more specific about what information they need.")
        ]
        
        # Add conversation history
        for msg in request.conversation_history[-20:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(SystemMessage(content=msg.content))
        
        # Add current message
        messages.append(HumanMessage(content=request.message))
        
        response = await model.ainvoke(messages)
        
        return ChatResponse(
            response=response.content,
            tools_used=["general_chat"]
        )
        
    except Exception as e:
        logger.error(f"❌ General chat error: {e}")
        return ChatResponse(
            response="I'm having trouble processing your request right now. Please try again.",
            tools_used=["error"]
        )

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with better query processing"""
    return await enhanced_chat_processing(request)

@app.post("/query", response_model=ChatResponse)
async def direct_query(request: QueryRequest):
    """Execute a direct SQL SELECT query"""
    try:
        if not request.query.strip().upper().startswith('SELECT'):
            return ChatResponse(
                response="Only SELECT queries are allowed",
                tools_used=["error"]
            )
        
        client = await get_mcp_client()
        if client:
            query_result = await client.call_tool("execute_query", {"query": request.query})
            result = await process_mcp_result(query_result)
        else:
            result = await execute_direct_query(request.query)
        
        if isinstance(result, dict) and result.get("error"):
            return ChatResponse(
                response=f"Query error: {result['error']}",
                tools_used=["error"]
            )
        
        if result.get("rows"):
            formatted_result = format_query_results(result["rows"], limit=10)
            row_count = result.get("row_count", len(result["rows"]))
            
            if row_count > 10:
                formatted_result += f"\n\n... and {row_count - 10} more rows (showing first 10)"
            
            return ChatResponse(
                response=formatted_result,
                tools_used=["direct_sql_query"]
            )
        else:
            return ChatResponse(
                response="No results found for your query.",
                tools_used=["direct_sql_query"]
            )
        
    except Exception as e:
        logger.error(f"❌ Direct query error: {e}")
        return ChatResponse(
            response="Error executing query. Please check your SQL syntax and try again.",
            tools_used=["error"]
        )

@app.get("/schema")
async def get_schema():
    """Get cached database schema"""
    try:
        if not schema_cache["last_updated"]:
            await build_enhanced_schema_cache()
        return schema_cache
    except Exception as e:
        logger.error(f"❌ Schema retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving schema")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_ok = await test_database_connection()
        mcp_ok = (await get_mcp_client()) is not None
        return {
            "status": "healthy" if db_ok else "unhealthy",
            "database": "connected" if db_ok else "disconnected",
            "mcp": "connected" if mcp_ok else "disconnected",
            "schema_cache": "loaded" if schema_cache["last_updated"] else "not loaded"
        }
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "database": "error",
            "mcp": "error",
            "schema_cache": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
