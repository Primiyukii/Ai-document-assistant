from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db

app = FastAPI(
    title="AI Document Assistant API",
    description="RAG-based document Q&A assistant API",
    version="0.1.0",
)

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Document Assistant API"}

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
