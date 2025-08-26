import os
import logging
import uuid
from uuid import UUID
from datetime import datetime
from dotenv import load_dotenv
from fastapi import Path, FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from sqlalchemy.orm import Session

from .database import SessionLocal, engine, Base
from .models import Session as DBSession, Message
from .vectorstore import ingest_chunks, query_chunks
from .retrieval import retrieve_context
from .generation import generate_medical_answer, generate_fallback_response

from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)
# Create database tables
# Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="mediCore AI Chatbot API",
    description="FastAPI backend for mediCore RAG-based medical chatbot",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#mount static files here 
app.mount("/static", StaticFiles(directory="static"), name="static")

def check_env():
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable not set")
        raise RuntimeError("GOOGLE_API_KEY not set")
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable not set")
        raise RuntimeError("DATABASE_URL not set")
    
@app.on_event("startup")
async def on_startup():
    check_env()
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Schemas
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


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {
        "status": "ok",
        "service": "mediCore AI Chatbot",
        "version": app.version,
        "gemini_configured": bool(os.getenv("GOOGLE_API_KEY")),
    }


# Ingest endpoint
@app.post("/ingest", tags=["Vector Store"], summary="Ingest document chunks")
async def ingest_documents(chunks: List[DocumentChunk]):
    """
    Ingest a list of document chunks: generate embeddings and store them in ChromaDB.
    """
    if not chunks:
        raise HTTPException(status_code=400, detail="No document chunks provided")
    ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [{"source": c.source} for c in chunks]
    try:
        ingest_chunks(ids=ids, texts=texts, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return {"message": f"Successfully ingested {len(chunks)} chunks."}


# Query endpoint
@app.get("/query", tags=["Vector Store"], summary="Query similar document chunks")
async def query_documents(
    query: str = Query(..., description="Text to search against stored chunks"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top results to return"),
):
    """
    Query the vector store for the most similar document chunks to the input text.
    """
    try:
        results = query_chunks(query_text=query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    hits = []
    for idx, doc_id in enumerate(results["ids"][0]):
        hits.append(
            {
                "id": doc_id,
                "text": results["documents"][0][idx],
                "source": results["metadatas"][0][idx].get("source"),
                "distance": results["distances"][0][idx],
            }
        )
    return {"query": query, "results": hits}


# Chat endpoint
@app.post("/chat", response_model=ChatResponse, tags=["Chat"], summary="RAG-powered medical chat")
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Receive a user question, retrieve context, generate an AI answer, persist conversation, and return response.
    """
    start = datetime.utcnow()

    # Session handling
    session_uuid = req.session_uuid or str(uuid.uuid4())
    db_sess = db.query(DBSession).filter_by(session_uuid=session_uuid).first()
    if not db_sess:
        db_sess = DBSession(session_uuid=session_uuid)
        db.add(db_sess)
        db.commit()
        db.refresh(db_sess)

    # Retrieve context with error handling
    try:
        context_chunks = retrieve_context(req.question, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {e}")

    # Generate answer with error handling
    try:
        if context_chunks:
            answer = generate_medical_answer(req.question, context_chunks)
        else:
            answer = generate_fallback_response(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e}")

    elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)

    # Persist user and assistant messages
    user_msg = Message(
        session_id=db_sess.id,
        role="user",
        content=req.question,
        token_count=None,
        response_time_ms=None,
    )
    db.add(user_msg)
    db.flush()  # ensure user_msg.id is set if needed

    bot_msg = Message(
        session_id=db_sess.id,
        role="assistant",
        content=answer,
        token_count=None,
        response_time_ms=elapsed,
    )
    db.add(bot_msg)
    db.commit()

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
async def get_chat_history(session_uuid: UUID = Path(..., description="The UUID of the chat session"), db: Session = Depends(get_db)):
    """
    Retrieve the full chat history for a given session UUID.
    """
    # Look up the session
    db_sess = db.query(DBSession).filter_by(session_uuid=session_uuid).first()
    if not db_sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch messages ordered by timestamp
    messages = (
        db.query(Message)
          .filter_by(session_id=db_sess.id)
          .order_by(Message.timestamp.asc())
          .all()
    )

    # Build a serializable response
    history = [
        {
            "sender": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
        }
        for msg in messages
    ]

    return {"session_uuid": str(session_uuid), "messages": history}