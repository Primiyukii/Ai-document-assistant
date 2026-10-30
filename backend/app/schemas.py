from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List, Optional

class ChunkBase(BaseModel):
    page_number: int
    chunk_index: int
    content: str

class ChunkResponse(ChunkBase):
    id: int
    document_id: int
    similarity: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    file_size: Optional[int] = None

class DocumentResponse(DocumentBase):
    id: int
    file_size: Optional[int]
    upload_date: datetime
    chunk_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)

class DocumentDetailResponse(DocumentResponse):
    chunks: List[ChunkResponse] = []

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[ChunkResponse]
