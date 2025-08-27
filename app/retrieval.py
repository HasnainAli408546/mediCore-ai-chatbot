import logging
from typing import List, Dict
from dotenv import load_dotenv
from .vectorstore import query_chunks

# Load environment variables and configure logging
load_dotenv()
logger = logging.getLogger(__name__)

def retrieve_context(
    question: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Given a user question, retrieve the top_k most similar document chunks.
    Returns a list of dicts with id, text, source, distance, and relevance_score.
    """
    logger.info(f"Retrieving context for query: '{question[:50]}...', top_k={top_k}")
    
    try:
        # Perform the vector similarity query
        logger.info("Querying vector store for similar documents...")
        results = query_chunks(query_text=question, top_k=top_k)
        
        if not results or not results.get("ids") or not results["ids"][0]:
            logger.warning("No results returned from vector store")
            return []
        
        logger.info(f"Vector store returned {len(results['ids'][0])} potential matches")
        
        hits: List[Dict] = []
        for idx, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx]
            relevance_score = round(1.0 - distance, 3)
            
            hits.append({
                "id": doc_id,
                "text": results["documents"][0][idx],
                "source": results["metadatas"][0][idx].get("source", "unknown"),
                "distance": distance,
                # Convert distance to a simple relevance score (higher is better)
                "relevance_score": relevance_score
            })
        
        # Sort by ascending distance (i.e., descending relevance_score)
        hits.sort(key=lambda x: x["distance"])
        
        logger.info(f"Successfully retrieved {len(hits)} context chunks")
        if hits:
            best_score = hits[0]["relevance_score"]
            worst_score = hits[-1]["relevance_score"]
            logger.info(f"Relevance scores range: {worst_score} to {best_score}")
        
        return hits
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []

def format_context_for_prompt(context_chunks: List[Dict]) -> str:
    """
    Format the retrieved chunks into a single string for prompting.
    """
    logger.info(f"Formatting {len(context_chunks)} context chunks for prompt")
    
    if not context_chunks:
        logger.warning("No context chunks provided for formatting")
        return "No relevant medical context found."

    sections: List[str] = []
    total_chars = 0
    
    for i, chunk in enumerate(context_chunks, start=1):
        section = (
            f"[Source {i}: {chunk['source']}]\n"
            f"{chunk['text']}\n"
            f"(Relevance: {chunk['relevance_score']})"
        )
        sections.append(section)
        total_chars += len(section)
    
    formatted_context = "\n\n---\n\n".join(sections)
    logger.info(f"Formatted context: {len(sections)} sections, {total_chars} total characters")
    
    return formatted_context
