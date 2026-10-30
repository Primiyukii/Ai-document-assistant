from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import SearchRequest, ChatRequest, ChatResponse, ChunkResponse
from app.services.embedding import generate_embedding
from app.services.rag_service import search_similar_chunks, answer_question

router = APIRouter(prefix="/api/rag", tags=["RAG Search & Q&A"])

@router.post("/search", response_model=list[ChunkResponse])
def semantic_search(request: SearchRequest, db: Session = Depends(get_db)):
    """
    Performs semantic vector search across document chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_vector = generate_embedding(request.query)
    scored_chunks = search_similar_chunks(db, query_vector, top_k=request.top_k)

    results = []
    for chunk, similarity in scored_chunks:
        results.append(
            ChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                similarity=round(similarity, 4)
            )
        )
    return results

@router.post("/chat", response_model=ChatResponse)
def chat_qa(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Answers user questions using RAG (retrieval + Claude generation).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer, scored_chunks = answer_question(
        db=db,
        question=request.question,
        top_k=request.top_k,
        system_prompt=request.system_prompt
    )

    sources = []
    for chunk, similarity in scored_chunks:
        sources.append(
            ChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                similarity=round(similarity, 4)
            )
        )

    return ChatResponse(
        question=request.question,
        answer=answer,
        sources=sources
    )
