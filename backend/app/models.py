import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship to chunks (cascade delete so deleting a doc removes its chunks)
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    # Vector embedding column. We'll use 384 dimensions for sentence-transformers (e.g., all-MiniLM-L6-v2)
    embedding = Column(Vector(384), nullable=True)

    # Relationship to document
    document = relationship("Document", back_populates="chunks")
