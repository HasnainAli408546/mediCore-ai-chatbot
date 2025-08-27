import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load environment variables and configure logging
load_dotenv()
logger = logging.getLogger(__name__)

# Load embedding model once
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

try:
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}")
    raise

# Initialize ChromaDB client (persistent local storage)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chromadb_data")
logger.info(f"Initializing ChromaDB at path: {CHROMA_DB_PATH}")

try:
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    _collection = _chroma_client.get_or_create_collection(
        name="clinical_guidelines",
        metadata={"description": "Clinical guideline text chunks for mediCore RAG chatbot"}
    )
    logger.info("ChromaDB client and collection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.
    """
    logger.info(f"Generating embeddings for {len(texts)} text chunks")
    
    try:
        # SentenceTransformer returns a numpy array, convert to list
        embeddings = _embedding_model.encode(texts, show_progress_bar=True)
        embedding_list = embeddings.tolist()
        
        logger.info(f"Successfully generated {len(embedding_list)} embeddings, "
                   f"dimension: {len(embedding_list[0]) if embedding_list else 0}")
        return embedding_list
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def ingest_chunks(
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict]
) -> None:
    """
    Add new text chunks (with embeddings and metadata) to ChromaDB.
    """
    logger.info(f"Ingesting {len(texts)} document chunks into vector store")
    logger.info(f"Document IDs: {ids[:3]}{'...' if len(ids) > 3 else ''}")
    
    try:
        embeddings = embed_texts(texts)
        
        _collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # Get collection count after ingestion
        collection_count = _collection.count()
        logger.info(f"Vector store ingestion completed successfully. "
                   f"Collection now contains {collection_count} documents")
        
    except Exception as e:
        logger.error(f"Error during vector store ingestion: {e}")
        raise

def query_chunks(
    query_text: str,
    top_k: int = 3
) -> Dict:
    """
    Retrieve the top_k most similar document chunks for a query.
    """
    logger.info(f"Querying vector store with: '{query_text[:50]}...', top_k={top_k}")
    
    try:
        query_embedding = embed_texts([query_text])
        
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        num_results = len(results['ids'][0]) if results and results.get('ids') else 0
        logger.info(f"Vector store query returned {num_results} results")
        
        if num_results > 0:
            best_distance = min(results['distances'][0])
            worst_distance = max(results['distances'][0])
            logger.info(f"Distance range: {best_distance:.3f} to {worst_distance:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        raise
