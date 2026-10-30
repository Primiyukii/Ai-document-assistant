from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db
from app.routers import documents, rag

app = FastAPI(
    title="AI Document Assistant API",
    description="RAG-based document Q&A assistant API powered by PostgreSQL, pgvector, and Claude",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(rag.router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to AI Document Assistant API",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"

    return {
        "status": "healthy",
        "service": "ai-document-assistant-backend",
        "database": db_status
    }
