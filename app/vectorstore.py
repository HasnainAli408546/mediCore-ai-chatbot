import os
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load embedding model once
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB client (persistent local storage)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chromadb_data")
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection = _chroma_client.get_or_create_collection(
    name="clinical_guidelines",
    metadata={"description": "Clinical guideline text chunks for mediCore RAG chatbot"}
)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.
    """
    # SentenceTransformer returns a numpy array, convert to list
    embeddings = _embedding_model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

def ingest_chunks(
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict]
) -> None:
    """
    Add new text chunks (with embeddings and metadata) to ChromaDB.
    """
    embeddings = embed_texts(texts)
    _collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_chunks(
    query_text: str,
    top_k: int = 3
) -> Dict:
    """
    Retrieve the top_k most similar document chunks for a query.
    """
    query_embedding = embed_texts([query_text])
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results
