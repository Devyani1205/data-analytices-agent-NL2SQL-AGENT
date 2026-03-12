import os
import logging
import json
import time
from datetime import datetime
from uuid import uuid4
from contextlib import asynccontextmanager


from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
from textwrap import dedent



import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.postgres import PostgresTools
from agno.storage.postgres import PostgresStorage
from dotenv import load_dotenv


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sqlagent.env')
if not os.path.exists(env_path):
    logger.error(f"Environment file not found at {env_path}")
    raise FileNotFoundError(f"Environment file not found at {env_path}")
load_dotenv(env_path)

DB_NAME = os.getenv("DB_NAME", "stagedatabase")
DB_USER = os.getenv("DB_USER", "master_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_postgres_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


connection_pool = None
knowledge_base = None
session_storage = None
active_sessions = {}

WORKING_TABLES = [
    'case_volume_by_speciality_mom',
    'case_volume_by_surgeon_mom',
    'case_volume_by_case_type_mom',
    'contribution_margin_per_case_monthly',
    'contribution_margin_per_surgeon_monthly',
    'contribution_margin_per_case_type_monthly',
    'payer_mix_andreimbursement_trends',
    'block_time_utilization_by_surgeon',
    'first_case_on_time_start_by_surgeon',
    'first_case_on_time_starts_by_case_type',
    'turnover_time_between_case_by_surgeon_monthly',
    'turnover_time_between_cases_by_case_types_monthly',
    'physician_tat_overall',
    'physician_tat_by_surgeon_monthly',
    'physician_tat_by_case_type_monthly',
    'case_goes_beyond_block_by_surgeon',
    'average_end_time_of_last_case_wrt_block_time',
    'surgeon_case_volume_per_payer_type',
    'surgeon_profitability_by_payer_type',
    'serviceline_case_mix_and_profitability'
]


class KnowledgeBase:
    """PostgreSQL-based knowledge base for storing conversations"""
    
    def __init__(self, db_url=DB_URL):
        self.db_url = db_url
        self.init_db()
    
    def init_db(self):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
               
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_conversations (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        result_count INTEGER,
                        success BOOLEAN,
                        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                        execution_time REAL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        response_data JSONB DEFAULT '{}'::jsonb,
                        conversation_id TEXT
                    )
                """)
                
               
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS active_conversations (
                        user_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_conversations_user_id 
                    ON user_conversations(user_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_conversations_conversation_id 
                    ON user_conversations(conversation_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_conversations_timestamp 
                    ON user_conversations(timestamp DESC)
                """)
                
                conn.commit()
                logger.info("Initialized knowledge base database")
    
    def get_connection(self):
        from contextlib import contextmanager
        @contextmanager
        def _get_conn():
            conn = psycopg2.connect(self.db_url.replace("postgresql+psycopg", "postgresql"))
            try:
                yield conn
            finally:
                conn.close()
        return _get_conn()
    
    def create_new_conversation_for_user(self, user_id):
        """Create a new conversation and set it as active for the user"""
        conversation_id = str(uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute("""
                        INSERT INTO active_conversations (user_id, conversation_id, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (user_id) 
                        DO UPDATE SET conversation_id = EXCLUDED.conversation_id, 
                                      updated_at = NOW()
                    """, (user_id, conversation_id))
                    
                    conn.commit()
                    logger.info(f"Created new conversation {conversation_id} for user {user_id}")
                    return conversation_id
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to create new conversation: {str(e)}")
                    raise
    
    def set_active_conversation(self, user_id, conversation_id):
        """Set a specific conversation as active for the user"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM user_conversations 
                        WHERE conversation_id = %s AND user_id = %s
                    """, (conversation_id, user_id))
                    
                    if cursor.fetchone()[0] == 0:
                        raise ValueError(f"Conversation {conversation_id} not found for user {user_id}")
                    
                    # Set as active
                    cursor.execute("""
                        INSERT INTO active_conversations (user_id, conversation_id, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (user_id) 
                        DO UPDATE SET conversation_id = EXCLUDED.conversation_id, 
                                      updated_at = NOW()
                    """, (user_id, conversation_id))
                    
                    conn.commit()
                    logger.info(f"Set conversation {conversation_id} as active for user {user_id}")
                    return conversation_id
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to set active conversation: {str(e)}")
                    raise
    
    def get_or_create_active_conversation(self, user_id):
        """Get the active conversation for a user, or create one if none exists"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT conversation_id FROM active_conversations 
                    WHERE user_id = %s
                """, (user_id,))
                
                result = cursor.fetchone()
                if result:
                    return result[0]
                return self.create_new_conversation_for_user(user_id)
    
    def get_user_conversations_grouped(self, user_id):
        """Get all conversations for a user, grouped by conversation_id"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        conversation_id,
                        MIN(timestamp) as first_message_time,
                        MAX(timestamp) as last_message_time,
                        COUNT(*) as message_count,
                        STRING_AGG(query, ' | ' ORDER BY timestamp) as preview_queries
                    FROM user_conversations
                    WHERE user_id = %s AND conversation_id IS NOT NULL
                    GROUP BY conversation_id
                    ORDER BY MAX(timestamp) DESC
                """, (user_id,))
                
                return cursor.fetchall()
    
    def get_conversation_messages(self, conversation_id):
        """Get all messages in a specific conversation"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, user_id, query, result_count, execution_time, success, 
                           timestamp, metadata, response_data, conversation_id
                    FROM user_conversations
                    WHERE conversation_id = %s
                    ORDER BY timestamp ASC
                """, (conversation_id,))
                
                return cursor.fetchall()
    
    def delete_conversation(self, conversation_id, user_id):
        """Delete all messages in a conversation"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Delete messages
                    cursor.execute("""
                        DELETE FROM user_conversations 
                        WHERE conversation_id = %s AND user_id = %s
                    """, (conversation_id, user_id))
                    
                    deleted_count = cursor.rowcount
                    
                    # Remove from active conversations if it was active
                    cursor.execute("""
                        DELETE FROM active_conversations 
                        WHERE user_id = %s AND conversation_id = %s
                    """, (user_id, conversation_id))
                    
                    conn.commit()
                    logger.info(f"Deleted conversation {conversation_id} for user {user_id} ({deleted_count} messages)")
                    return deleted_count
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to delete conversation: {str(e)}")
                    raise
    
    def store_conversation(self, user_id, query, result_count=0, success=True,
                          execution_time=0, metadata=None, response_data=None, conversation_id=None):
        """Store conversation"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # If no conversation_id provided, get or create active one
                    if not conversation_id:
                        conversation_id = self.get_or_create_active_conversation(user_id)
                    
                    insert_data = {
                        'user_id': str(user_id),
                        'query': query.strip(),
                        'result_count': int(result_count),
                        'success': bool(success),
                        'execution_time': float(execution_time),
                        'conversation_id': conversation_id
                    }
                    
                    if metadata is None:
                        metadata = {}
                    insert_data['metadata'] = json.dumps(metadata, ensure_ascii=False)
                    
                    if response_data is None:
                        response_data = {'data': []}
                    insert_data['response_data'] = json.dumps(response_data, ensure_ascii=False)
                    
                    cursor.execute("""
                        INSERT INTO user_conversations 
                        (user_id, query, result_count, success, execution_time, metadata, response_data, conversation_id)
                        VALUES (%(user_id)s, %(query)s, %(result_count)s, %(success)s, %(execution_time)s, %(metadata)s, %(response_data)s, %(conversation_id)s)
                        RETURNING id, conversation_id
                    """, insert_data)
                    
                    result = cursor.fetchone()
                    conn.commit()
                    return result[0], result[1]
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to store conversation: {str(e)}")
                    raise
    
    def get_user_conversations(self, user_id, limit=10, offset=0):
        """Retrieve paginated conversations"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, user_id, query, result_count, execution_time, success, 
                           timestamp, metadata, response_data, conversation_id
                    FROM user_conversations
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                """, (user_id, limit, offset))
                
                conversations = []
                for row in cur.fetchall():
                    metadata = {}
                    if row['metadata']:
                        try:
                            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse metadata: {e}")
                    
                    response_data = row.get('response_data', {})
                    if isinstance(response_data, str):
                        try:
                            response_data = json.loads(response_data)
                        except (json.JSONDecodeError, TypeError):
                            response_data = {'data': []}
                    
                    timestamp = row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp']
                    
                    conversation = {
                        'id': row['id'],
                        'user_id': row['user_id'],
                        'query': row['query'],
                        'result_count': row['result_count'],
                        'execution_time': row['execution_time'],
                        'success': row['success'],
                        'timestamp': timestamp,
                        'metadata': metadata,
                        'response_data': response_data,
                        'conversation_id': row.get('conversation_id'),
                        'chat_history': [
                            {
                                "role": "user",
                                "content": row['query'],
                                "timestamp": timestamp,
                                "source": "knowledge_base",
                                "conversation_id": row['id']
                            }
                        ]
                    }
                    
                    # Process response data for display
                    html_parts = []
                    for item in response_data.get('data', []):
                        if item.get('type') == 'text' and 'content' in item and 'html' in item['content']:
                            html_parts.append(item['content']['html'])
                    
                    conversation['chat_history'].append({
                        "role": "assistant",
                        "content": "\n".join(html_parts) if html_parts else "<p>No response content available</p>",
                        "timestamp": timestamp,
                        "source": "knowledge_base",
                        "conversation_id": row['id'],
                        "result_count": row['result_count'],
                        "success": row['success'],
                        "metadata": metadata
                    })
                    
                    conversations.append(conversation)
                
                return conversations


def get_connection_pool():
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=60,
            application_name="MedicalAnalyticsAgent"
        )
        logger.info("Connection pool created successfully")
        return connection_pool
    except Exception as e:
        logger.error(f"Connection pool creation failed: {e}")
        raise ValueError(f"Failed to create connection pool: {e}")

async def sql_agent(query: str, user_id: str = None, conversation_id: str = None):
    """Enhanced analytics agent with textual formatting"""
    start_time = time.time()
    logger.info(f"Processing query for user {user_id}, conversation {conversation_id}: {query}")
    
    try:
        if not conversation_id:
            conversation_id = knowledge_base.get_or_create_active_conversation(user_id)
        
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Empty query provided",
                "conversation_id": conversation_id,
                "data": [{"type": "text", "content": {"html": "<p>Please provide a valid query.</p>"}}]
            }
        
        
        kpi_formulas = dedent("""
            **KPI CALCULATION FORMULAS:**
            
            1. **Case Volume by Specialty**: Count of completed cases where Specialty = selected
            2. **Case Volume by Surgeon**: Count of completed cases where Surgeon = selected
            3. **Case Volume by Case Type**: Count of completed cases where Case Type = selected
            
            4. **Contribution Margin per Case**: Net Revenue − (Variable Supply Cost + Implant Cost + Variable Labor Cost) [per case]
            5. **Contribution Margin per Surgeon**: Sum of (Net Revenue − Variable Supply Cost − Implant Cost − Variable Labor Cost) for the surgeon
            6. **Contribution Margin per Case Type**: (Total Net Revenue for the Case Type − Total Variable Supply Costs − Total Implant Costs − Total Variable Labor Costs) ÷ Number of Cases of that Case Type
            
            7. **Payer Mix**: (Net Revenue by Payor ÷ Total Net Revenue) tracked by period (and/or Avg Net Revenue per Case by Payor)
            
            8. **Block Time Utilization**: Scheduled Minutes Used ÷ Scheduled Block Minutes [per surgeon]
            9. **Actual OR Time vs Block Time**: Actual OR Minutes (wheels-in first case → wheels-out last case) ÷ Scheduled Block Minutes [per surgeon]
            
            10. **First Case On-Time Start (by Surgeon)**: (Number of first cases with Actual Wheels-In ≤ Scheduled Wheels-In) ÷ (Total first cases) [by surgeon]
            11. **First Case On-Time Start (by Case Type)**: (Number of first cases on time) ÷ (Total first cases) [by case type]
            
            12. **Turnover Time by Surgeon**: Average(Wheels-In of case N+1 − Wheels-Out of case N) in minutes [by surgeon]
            13. **Turnover Time by Case Type**: Average(Wheels-In of case N+1 − Wheels-Out of case N) in minutes [by case type]
            
            14. **Physician TAT Overall**: Average(Next Case Incision Time − Prior Case Surgeon Exit Time) in minutes [overall]
            15. **Physician TAT by Surgeon**: Average(Next Case Incision Time − Prior Case Surgeon Exit Time) in minutes [by surgeon]
            16. **Physician TAT by Case Type**: Average(Next Case Incision Time − Prior Case Surgeon Exit Time) in minutes [by case type]
            
            17. **Cases Beyond Block Time**: (Days where Actual Last Case Out > Scheduled Block End) ÷ (Total days with block time) [per surgeon]
            18. **Average End Time vs Block Time**: Average(Actual Last Case Out − Scheduled Block End) in minutes [positive = overrun; negative = underrun]
            
            19. **Surgeon Case Volume per Payer**: Count of completed cases per surgeon per payor type
            20. **Surgeon Profitability by Payer**: Sum over surgeon & payor of (Net Revenue − Variable Supply − Implant − Variable Labor) − Allocated Fixed Costs [filtered by payor]
            
            21. **Service Line Case Mix & Profitability**: 
                - Case Mix % = Cases in Service Line ÷ Total Cases
                - Profitability = (Revenue − Variable Supply − Implant − Variable Labor) − Allocated Fixed Costs [by service line]
        """)
        
    
        analytics_instructions = dedent(f"""
            You are a healthcare analytics expert.
            
            **CRITICAL: Format your ENTIRE response as clean HTML with proper structure.**
            
            **RESPONSE STRUCTURE (use HTML tags):**
            
            <h3>Executive Summary</h3>
            <p>Key findings in 2-3 sentences</p>
            
            <h3>Data Analysis</h3>
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <thead>
                    <tr style="background: #f8fafc; color: black; border-bottom: 2px solid #e2e8f0;">
                        <th style="padding: 12px; border: 1px solid #ddd;">Column</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">Value</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>Key Metrics</h3>
            <ul>
                <li>Metric 1: value</li>
                <li>Metric 2: value</li>
            </ul>
            
            <h3>Insights & Distribution</h3>
            <p>Provide insights, trends, patterns, and distribution analysis here.</p>
            
            <h3>Recommendations</h3>
            <ol>
                <li>Recommendation 1</li>
                <li>Recommendation 2</li>
            </ol>
            
            <h3>Follow-up Questions</h3>
            <p>Based on this analysis, you might want to explore:</p>
            <ul>
                <li>Follow-up question 1?</li>
                <li>Follow-up question 2?</li>
                <li>Follow-up question 3?</li>
            </ul>
            
            **AVAILABLE TABLES:** {', '.join(WORKING_TABLES)}
            {kpi_formulas}
        """)
       
        postgres_tools = PostgresTools(
            host=DB_HOST, port=int(DB_PORT), db_name=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
        )
        
        agent = Agent(
            name="Healthcare Analytics Agent",
            model=Groq(id="openai/gpt-oss-120b"),
            tools=[postgres_tools],  
            storage=session_storage,
            description="Expert healthcare analytics agent",
            instructions=analytics_instructions,
            markdown= False  # this will Set to False for HTML output
        )
        
        # Run agent
        result = agent.run(query, timeout=90, user_id=user_id, session_id=conversation_id)
        execution_time = time.time() - start_time
        
        # Build response
        response_data = {
            "success": True,
            "query": query,
            "execution_time": execution_time,
            "conversation_id": conversation_id,
            "data": []
        }
        
        # # Add formatted text content
        # html_content = f"""
        # <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        #              padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;
        #             box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
        #     <h2 style='margin: 0 0 15px 0; font-size: 28px;'> Analytics Results</h2>
        #     <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
        #         {result.content}
        #     </div>
        # </div>
        # """
        
        html_content = f"""
        <div style='background: white; padding: 30px; border-radius: 15px; color: black; margin-bottom: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;'>
            <h2 style='margin: 0 0 15px 0; font-size: 28px; color: black;'>Analytics Results</h2>
            <div style='background: white; padding: 20px; border-radius: 10px; color: black;'>
                {result.content}
            </div>
        </div>
        """
        
        response_data["data"].append({
            "type": "text",
            "content": {"html": html_content}
        })
        
        # Store conversation
        if user_id and knowledge_base:
            knowledge_base.store_conversation(
                user_id=user_id, query=query, conversation_id=conversation_id,
                result_count=len(response_data["data"]), success=True,
                execution_time=execution_time, response_data=response_data
            )
        
        logger.info(f"Query processed in {execution_time:.2f}s - Conversation: {conversation_id}")
        return response_data
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Query failed after {error_time:.2f}s: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "conversation_id": conversation_id,
            "execution_time": error_time,
            "data": [{"type": "text", "content": {"html": f"<p><strong>Error:</strong> {str(e)}</p>"}}]
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection_pool, knowledge_base, session_storage
    try:
        knowledge_base = KnowledgeBase(db_url=DB_URL)
        session_storage = PostgresStorage(
            table_name="agent_sessions",
            db_url=DB_URL
        )
        get_connection_pool()
        logger.info("Application startup completed")
        yield
    finally:
        if connection_pool:
            connection_pool.closeall()
            logger.info("Connection pool closed")
        logger.info("Application shutdown completed")


app = FastAPI(
    title="Medical Analytics Assistant",
    description="Natural Language Analytics System for Medical Practice Data",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)
socket_app = socketio.ASGIApp(sio, app)


@sio.event
async def connect(sid, environ):
    logger.info(f"Socket connected: {sid}")
    await sio.emit('connection_success', {'message': 'Connected to Analytics Agent'}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"Socket disconnected: {sid}")


@sio.event
async def query(sid, data):
    try:
        query = data.get('query')
        user_id = data.get('user_id')
        conversation_id = data.get('conversation_id')
        if not query:
            await sio.emit('query_error', {'error': 'Query is required'}, to=sid)
            return
        result = await sql_agent(query, user_id, conversation_id)
        await sio.emit('query_result', result, to=sid)
    except Exception as e:
        logger.error(f"Socket query error: {str(e)}")
        await sio.emit('query_error', {'error': str(e)}, to=sid)


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))


@app.post("/agent/api/query")
async def query_endpoint(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        user_id = data.get("user_id")
        conversation_id = data.get("conversation_id")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        result = await sql_agent(query, user_id, conversation_id)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/api/chat-history/{user_id}")
async def get_chat_history_api(user_id: str, page: int = 1, per_page: int = 20):
    try:
        offset = (page - 1) * per_page
        conversations = knowledge_base.get_user_conversations(user_id, limit=per_page, offset=offset)
        
        return {
            'success': True,
            'user_id': user_id,
            'chat_history': conversations,
            'pagination': {
                'page': page,
                'per_page': per_page
            }
        }
    except Exception as e:
        logger.error(f"API chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# New Conversation Management Endpoints
@app.post("/agent/api/new-conversation")
async def create_new_conversation(request: Request):
    """Create a new conversation for a user"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        conversation_id = knowledge_base.create_new_conversation_for_user(user_id)
        
        return JSONResponse(content={
            "success": True,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "message": "New conversation created and set as active"
        })
        
    except Exception as e:
        logger.error(f"New conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/api/switch-conversation")
async def switch_to_conversation(request: Request):
    """Switch to an existing conversation"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        conversation_id = data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise HTTPException(status_code=400, detail="User ID and Conversation ID are required")
        
        active_conversation_id = knowledge_base.set_active_conversation(user_id, conversation_id)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "active_conversation_id": active_conversation_id,
            "message": f"Switched to conversation {conversation_id}"
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Switch conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/api/active-conversation/{user_id}")
async def get_active_conversation(user_id: str):
    """Get user's current active conversation_id"""
    try:
        conversation_id = knowledge_base.get_or_create_active_conversation(user_id)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "active_conversation_id": conversation_id
        })
        
    except Exception as e:
        logger.error(f"Get active conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/api/conversations/{user_id}")
async def get_user_conversations_list(user_id: str):
    """Get list of conversations for a user"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
            
        conversations = knowledge_base.get_user_conversations_grouped(user_id)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "conversations": [
                {
                    "conversation_id": str(conv['conversation_id']),
                    "first_message_time": conv['first_message_time'].isoformat() if conv['first_message_time'] else None,
                    "last_message_time": conv['last_message_time'].isoformat() if conv['last_message_time'] else None,
                    "message_count": conv['message_count'],
                    "preview": (conv['preview_queries'][:100] + "...") if conv['preview_queries'] and len(str(conv['preview_queries'])) > 100 else str(conv['preview_queries'])
                }
                for conv in conversations
            ]
        })
    except Exception as e:
        logger.error(f"Get conversations error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to load conversations",
                "message": str(e)
            }
        )

@app.get("/agent/api/conversation/{conversation_id}/messages")
async def get_conversation_messages_api(conversation_id: str):
    """Get all messages in a specific conversation"""
    try:
        messages = knowledge_base.get_conversation_messages(conversation_id)
        
        chat_history = []
        
        for msg in messages:
            # Parse timestamp
            timestamp = msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']
            
            # Add user message
            chat_history.append({
                "role": "user",
                "content": msg['query'],
                "timestamp": timestamp,
                "message_id": msg['id']
            })
            
            # Parse response data
            response_data = msg.get('response_data', {})
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except (json.JSONDecodeError, TypeError):
                    response_data = {'data': []}
            
            # Extract HTML content from response
            html_parts = []
            for item in response_data.get('data', []):
                if item.get('type') == 'text' and 'content' in item and 'html' in item['content']:
                    html_parts.append(item['content']['html'])
            
            # Add assistant message
            chat_history.append({
                "role": "assistant",
                "content": "\n".join(html_parts) if html_parts else "<p>No response content available</p>",
                "timestamp": timestamp,
                "message_id": msg['id'],
                "result_count": msg.get('result_count', 0),
                "success": msg.get('success', True),
                "execution_time": msg.get('execution_time', 0)
            })
        
        return JSONResponse(content={
            "success": True,
            "conversation_id": conversation_id,
            "chat_history": chat_history,
            "message_count": len(messages)
        })
        
    except Exception as e:
        logger.error(f"Get conversation messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agent/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str = Query(...)):
    """Delete an entire conversation"""
    try:
        deleted_count = knowledge_base.delete_conversation(conversation_id, user_id)
        
        return JSONResponse(content={
            "success": True,
            "conversation_id": conversation_id,
            "deleted_messages": deleted_count,
            "message": f"Conversation deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Delete conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting server on 127.0.0.1:8002")
    uvicorn.run(
        socket_app,
        host="127.0.0.1",
        port=8002,
        log_level="debug",
        access_log=True
    )
    
    
    #this code working well but need add folow of question