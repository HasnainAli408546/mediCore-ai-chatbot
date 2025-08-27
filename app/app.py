import os
import uuid
import logging
import sys
import time
import psutil  # NEW IMPORT
from uuid import UUID
from datetime import datetime
from dotenv import load_dotenv
from fastapi import Path, FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import text

from sqlalchemy.orm import Session

from .database import SessionLocal, engine, Base
from .models import Session as DBSession, Message
from .vectorstore import ingest_chunks, query_chunks
from .retrieval import retrieve_context
from .generation import generate_medical_answer, generate_fallback_response

from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Add file handler for production
        # logging.FileHandler("medicore.log") 
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Startup environment validation
def check_env():
    """Validate required environment variables at startup"""
    required_vars = ["GOOGLE_API_KEY", "DATABASE_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("All required environment variables are set")

app = FastAPI(
    title="mediCore AI Chatbot API",
    description="FastAPI backend for mediCore RAG-based medical chatbot",
    version="0.1.0",
)

# Track startup time for uptime calculation
startup_time = time.time()  # NEW LINE

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def on_startup():
    """Initialize database tables and validate environment after server startup"""
    try:
        logger.info("Starting mediCore AI Chatbot API...")
        check_env()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
        logger.info("mediCore AI Chatbot API startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Schemas (unchanged)
class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str

class ChatRequest(BaseModel):
    session_uuid: Optional[str] = None
    question: str
    include_context: bool = True

class ContextChunk(BaseModel):
    id: str
    text: str
    source: str
    relevance_score: float

class ChatResponse(BaseModel):
    answer: str
    session_uuid: str
    context: List[ContextChunk]
    response_time_ms: int

# ENHANCED Health check - REPLACES THE OLD ONE
@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint for all system components"""
    start_time = time.time()
    logger.info("Comprehensive health check requested")
    
    health_status = {
        "status": "ok",
        "service": "mediCore AI Chatbot",
        "version": app.version,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - startup_time),
        "checks": {},
        "system_info": {}
    }
    
    # 1. Check Google API key configuration
    try:
        health_status["checks"]["gemini_api_key"] = {
            "status": "ok" if os.getenv("GOOGLE_API_KEY") else "error",
            "configured": bool(os.getenv("GOOGLE_API_KEY"))
        }
    except Exception as e:
        logger.error(f"Gemini API key check failed: {e}")
        health_status["checks"]["gemini_api_key"] = {"status": "error", "error": str(e)}
    
    # 2. Check database connection
    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "ok",
            "connected": True,
            "url_configured": bool(os.getenv("DATABASE_URL"))
        }
        db.close()
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        health_status["checks"]["database"] = {
            "status": "error",
            "connected": False,
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # # 3. Check ChromaDB vector store
    try:
        from .vectorstore import _collection
        collection_count = _collection.count()
        health_status["checks"]["vector_store"] = {
            "status": "ok",
            "connected": True,
            "document_count": collection_count,
            "collection_name": "clinical_guidelines"
        }
        logger.info(f"Vector store check: {collection_count} documents in collection")
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        health_status["checks"]["vector_store"] = {
            "status": "error",
            "connected": False,
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 4. Check embedding model
    try:
        from .vectorstore import _embedding_model, EMBEDDING_MODEL_NAME
        # Test with a small embedding
        test_embedding = _embedding_model.encode(["health check test"])
        health_status["checks"]["embedding_model"] = {
            "status": "ok",
            "model_name": EMBEDDING_MODEL_NAME,
            "embedding_dimension": len(test_embedding[0]),
            "loaded": True
        }
    except Exception as e:
        logger.error(f"Embedding model health check failed: {e}")
        health_status["checks"]["embedding_model"] = {
            "status": "error",
            "loaded": False,
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 5. System metrics
    try:
        health_status["system_info"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('.').percent,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    except Exception as e:
        logger.warning(f"System metrics collection failed: {e}")
        health_status["system_info"] = {"error": str(e)}
    
    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)
    health_status["response_time_ms"] = response_time_ms
    
    logger.info(f"Health check completed: {health_status['status']} ({response_time_ms}ms)")
    return health_status

# Ingest endpoint with logging
@app.post("/ingest", tags=["Vector Store"], summary="Ingest document chunks")
async def ingest_documents(chunks: List[DocumentChunk]):
    """Ingest document chunks with comprehensive logging"""
    if not chunks:
        logger.warning("Ingest attempted with no chunks provided")
        raise HTTPException(status_code=400, detail="No document chunks provided")
    
    logger.info(f"Starting ingestion of {len(chunks)} document chunks")
    
    ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [{"source": c.source} for c in chunks]
    
    try:
        ingest_chunks(ids=ids, texts=texts, metadatas=metadatas)
        logger.info(f"Successfully ingested {len(chunks)} chunks")
        return {"message": f"Successfully ingested {len(chunks)} chunks."}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

# Query endpoint with logging
@app.get("/query", tags=["Vector Store"], summary="Query similar document chunks")
async def query_documents(
    query: str = Query(..., description="Text to search against stored chunks"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top results to return"),
):
    """Query vector store with logging"""
    logger.info(f"Vector store query received: '{query[:50]}...' (top_k={top_k})")
    
    try:
        results = query_chunks(query_text=query, top_k=top_k)
        logger.info(f"Query returned {len(results.get('ids', [[]]))} results")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    
    hits = []
    for idx, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][idx],
            "source": results["metadatas"][0][idx].get("source"),
            "distance": results["distances"][0][idx],
        })
    
    return {"query": query, "results": hits}

# Chat endpoint with comprehensive logging
@app.post("/chat", response_model=ChatResponse, tags=["Chat"], summary="RAG-powered medical chat")
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    """Chat endpoint with detailed logging"""
    start = datetime.utcnow()
    session_uuid = req.session_uuid or str(uuid.uuid4())
    
    logger.info(f"Chat request received - Session: {session_uuid}, Question: '{req.question[:100]}...'")

    # Session handling
    db_sess = db.query(DBSession).filter_by(session_uuid=session_uuid).first()
    if not db_sess:
        db_sess = DBSession(session_uuid=session_uuid)
        db.add(db_sess)
        db.commit()
        db.refresh(db_sess)
        logger.info(f"New session created: {session_uuid}")
    else:
        logger.info(f"Using existing session: {session_uuid}")

    # Retrieve context with error handling and logging
    try:
        logger.info("Retrieving context from vector store...")
        context_chunks = retrieve_context(req.question, top_k=5)
        logger.info(f"Retrieved {len(context_chunks)} context chunks")
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {e}")

    # Generate answer with error handling and logging
    try:
        logger.info("Generating AI response...")
        if context_chunks:
            answer = generate_medical_answer(req.question, context_chunks)
            logger.info("Answer generated using context")
        else:
            answer = generate_fallback_response(req.question)
            logger.info("Fallback answer generated (no context)")
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e}")

    elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
    logger.info(f"Chat response completed in {elapsed}ms")

    # Persist messages
    try:
        user_msg = Message(
            session_id=db_sess.id,
            role="user",
            content=req.question,
            token_count=None,
            response_time_ms=None,
        )
        db.add(user_msg)
        db.flush()

        bot_msg = Message(
            session_id=db_sess.id,
            role="assistant",
            content=answer,
            token_count=None,
            response_time_ms=elapsed,
        )
        db.add(bot_msg)
        db.commit()
        logger.info("Messages persisted to database")
    except Exception as e:
        logger.error(f"Message persistence failed: {e}")
        db.rollback()

    # Build response payload
    formatted_context: List[ContextChunk] = []
    if req.include_context:
        for chunk in context_chunks:
            text_snip = chunk["text"]
            if len(text_snip) > 500:
                text_snip = text_snip[:500] + "..."
            formatted_context.append(
                ContextChunk(
                    id=chunk["id"],
                    text=text_snip,
                    source=chunk["source"],
                    relevance_score=chunk["relevance_score"],
                )
            )

    return ChatResponse(
        answer=answer,
        session_uuid=session_uuid,
        context=formatted_context,
        response_time_ms=elapsed,
    )

@app.get("/sessions/{session_uuid}/history", tags=["Chat History"])
async def get_chat_history(
    session_uuid: UUID = Path(..., description="The UUID of the chat session"), 
    db: Session = Depends(get_db)
):
    """Get chat history with logging"""
    logger.info(f"Chat history requested for session: {session_uuid}")
    
    # Look up the session
    db_sess = db.query(DBSession).filter_by(session_uuid=session_uuid).first()
    if not db_sess:
        logger.warning(f"Session not found: {session_uuid}")
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch messages
    messages = (
        db.query(Message)
          .filter_by(session_id=db_sess.id)
          .order_by(Message.timestamp.asc())
          .all()
    )

    logger.info(f"Retrieved {len(messages)} messages for session {session_uuid}")

    # Build response
    history = [
        {
            "sender": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
        }
        for msg in messages
    ]

    return {"session_uuid": str(session_uuid), "messages": history}
