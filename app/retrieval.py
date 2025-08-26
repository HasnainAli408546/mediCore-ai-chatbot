from typing import List, Dict
from .vectorstore import query_chunks

def retrieve_context(
    question: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Given a user question, retrieve the top_k most similar document chunks.
    Returns a list of dicts with id, text, source, distance, and relevance_score.
    """
    # Perform the vector similarity query
    results = query_chunks(query_text=question, top_k=top_k)

    hits: List[Dict] = []
    for idx, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][idx]
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][idx],
            "source": results["metadatas"][0][idx].get("source", "unknown"),
            "distance": distance,
            # Convert distance to a simple relevance score (higher is better)
            "relevance_score": round(1.0 - distance, 3)
        })

    # Sort by ascending distance (i.e., descending relevance_score)
    hits.sort(key=lambda x: x["distance"])
    return hits

def format_context_for_prompt(context_chunks: List[Dict]) -> str:
    """
    Format the retrieved chunks into a single string for prompting.
    """
    if not context_chunks:
        return "No relevant medical context found."

    sections: List[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        sections.append(
            f"[Source {i}: {chunk['source']}]\n"
            f"{chunk['text']}\n"
            f"(Relevance: {chunk['relevance_score']})"
        )
    # Separate sections with a clear delimiter
    return "\n\n---\n\n".join(sections)
