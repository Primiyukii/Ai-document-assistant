import os
import logging
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models import Chunk, Document
from app.services.embedding import generate_embedding
import anthropic

logger = logging.getLogger(__name__)

def search_similar_chunks(db: Session, query_vector: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
    """
    Performs vector similarity search using pgvector cosine distance.
    Returns list of (Chunk, similarity_score).
    Cosine distance = 1 - cosine_similarity. Similarity = 1 - distance.
    """
    try:
        # Using pgvector's cosine distance operator (<=>)
        # distance = Chunk.embedding.cosine_distance(query_vector)
        # similarity = 1 - distance
        distance_expr = Chunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(Chunk, distance_expr.label("distance"))
            .order_by(distance_expr)
            .limit(top_k)
        )
        results = db.execute(stmt).all()

        scored_chunks = []
        for chunk, distance in results:
            similarity = 1.0 - float(distance) if distance is not None else 0.0
            scored_chunks.append((chunk, similarity))

        return scored_chunks
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []

def answer_question(db: Session, question: str, top_k: int = 5, system_prompt: str = None) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Retrieves relevant chunks and generates an answer using Anthropic Claude.
    """
    # 1. Generate query embedding
    query_vec = generate_embedding(question)

    # 2. Retrieve top_k similar chunks
    scored_chunks = search_similar_chunks(db, query_vec, top_k=top_k)

    # 3. Build context from retrieved chunks
    context_parts = []
    for idx, (chunk, score) in enumerate(scored_chunks, 1):
        doc_name = chunk.document.filename if chunk.document else f"Document {chunk.document_id}"
        context_parts.append(
            f"[Source {idx}] (File: {doc_name}, Page: {chunk.page_number}, Similarity: {score:.2f})\n{chunk.content}"
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

    # 4. Prepare prompt for Claude
    default_system = (
        "You are a helpful AI document assistant. Your task is to answer the user's question accurately "
        "based strictly on the provided context sources. If the answer cannot be found in the context, "
        "state that clearly without making up information. Cite your sources using [Source X] notation."
    )
    sys_prompt = system_prompt or default_system

    user_message = f"""Context information:
{context_text}

User Question: {question}

Please provide a comprehensive answer grounded in the sources above, including citations."""

    # 5. Call Anthropic Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        # Fallback simulation if API key is not configured
        answer = (
            f"[Anthropic API key not configured. Simulated RAG Response]\n\n"
            f"Based on the retrieved documents, here is what I found regarding '{question}':\n\n"
            f"{context_text[:500]}..."
        )
        return answer, scored_chunks

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5-5",  # or claude-3-5-sonnet-20241022
            max_tokens=1500,
            system=sys_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        answer = response.content[0].text
        return answer, scored_chunks
    except Exception as e:
        logger.error(f"Error calling Anthropic API: {e}")
        answer = f"Error generating answer with Claude API: {str(e)}"
        return answer, scored_chunks
