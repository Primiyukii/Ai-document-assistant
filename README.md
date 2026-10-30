# AI Document Assistant (RAG) 📚

An end-to-end Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **PostgreSQL (pgvector)**, **Sentence-Transformers**, **Anthropic Claude**, and **Streamlit**.

## 🌟 Key Features

1. **Multi-Format Ingestion**: Upload PDF, DOCX, TXT, and Markdown documents.
2. **Text Chunking & Vectorization**: Automatic overlapping sliding-window chunking and vector embedding generation using `all-MiniLM-L6-v2` (384 dimensions).
3. **PGVector Similarity Search**: High-performance cosine distance vector retrieval in PostgreSQL.
4. **Claude AI Chat & Q&A**: Answers user questions grounded in document context with accurate citations ([Source X]).
5. **Interactive Streamlit Web UI**: Easy document management, semantic search preview, and real-time chat interface.
6. **Docker Compose Orchestration**: Containerized database, FastAPI backend, and Streamlit frontend.

---

## 🏗️ Architecture Overview

```
 ┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────────┐
 │ Streamlit UI    │ ────> │ FastAPI Backend        │ ────> │ PostgreSQL          │
 │ (Port 8501)     │ <──── │ (Port 8000)            │ <──── │ + pgvector (5433)   │
 └─────────────────┘       └────────────────────────┘       └─────────────────────┘
                                       │
                                       ├─> SentenceTransformers (Embeddings)
                                       └─> Anthropic API (Claude Q&A)
```

---

## 🚀 Quick Start with Docker

1. **Clone & Configure Environment**:
   Copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY`:
   ```bash
   cp .env.example .env
   ```

2. **Start Services**:
   ```bash
   docker compose up --build
   ```

3. **Access Services**:
   - **Frontend UI**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI API & Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **PostgreSQL Database**: `localhost:5433`

---

## 🛠️ Local Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start API server
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## 📡 API Endpoints

- `POST /api/documents/upload` - Upload file (PDF, DOCX, TXT), chunk text & store embeddings.
- `GET /api/documents` - List all uploaded documents and chunk counts.
- `GET /api/documents/{id}` - View document details and chunk contents.
- `DELETE /api/documents/{id}` - Remove document and associated vector chunks.
- `POST /api/rag/search` - Perform vector similarity search.
- `POST /api/rag/chat` - Perform RAG Q&A with Claude API.
- `GET /health` - Database & service status check.

---

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 16 + `pgvector` extension
- **ORM & Migrations**: SQLAlchemy 2.0, Alembic
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Anthropic Claude API (`claude-sonnet-5-5` / `claude-3-5-sonnet`)
- **Frontend**: Streamlit
