import logging
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Document, Chunk
from app.schemas import DocumentResponse, DocumentDetailResponse, ChunkResponse
from app.services.document_processor import extract_text_from_file, chunk_text
from app.services.embedding import generate_embeddings_batch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a document (PDF, DOCX, TXT), extracts text, chunks it,
    generates embeddings, and stores it in the PostgreSQL database with pgvector.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    file_size = len(content)

    try:
        # 1. Extract text pages
        pages_text = extract_text_from_file(content, file.filename)
        if not pages_text:
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        # 2. Chunk text
        text_chunks = chunk_text(pages_text, chunk_size=500, chunk_overlap=50)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated from document.")

        # 3. Generate embeddings in batch
        chunk_texts = [c["content"] for c in text_chunks]
        embeddings = generate_embeddings_batch(chunk_texts)

        # 4. Save document to DB
        db_doc = Document(filename=file.filename, file_size=file_size)
        db.add(db_doc)
        db.flush()  # Get document ID

        # 5. Save chunks with embeddings
        for i, chunk_data in enumerate(text_chunks):
            vector = embeddings[i] if i < len(embeddings) else [0.0] * 384
            db_chunk = Chunk(
                document_id=db_doc.id,
                page_number=chunk_data["page_number"],
                chunk_index=chunk_data["chunk_index"],
                content=chunk_data["content"],
                embedding=vector
            )
            db.add(db_chunk)

        db.commit()
        db.refresh(db_doc)

        return DocumentResponse(
            id=db_doc.id,
            filename=db_doc.filename,
            file_size=db_doc.file_size,
            upload_date=db_doc.upload_date,
            chunk_count=len(text_chunks)
        )

    except ValueError as ve:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        db.rollback()
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """
    Lists all uploaded documents with chunk counts.
    """
    docs = db.query(Document).all()
    results = []
    for doc in docs:
        chunk_count = len(doc.chunks)
        results.append(
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_size=doc.file_size,
                upload_date=doc.upload_date,
                chunk_count=chunk_count
            )
        )
    return results

@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """
    Gets document details along with its chunks.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_responses = [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            content=c.content
        )
        for c in doc.chunks
    ]

    return DocumentDetailResponse(
        id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        upload_date=doc.upload_date,
        chunk_count=len(doc.chunks),
        chunks=chunk_responses
    )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Deletes a document and its associated chunks.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    return None
