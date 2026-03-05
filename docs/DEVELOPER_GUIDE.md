# Developer Guide — Hierarchical Legal RAG

This guide explains how the system works, where to make changes, and how to avoid common mistakes.
Written for developers joining the project — no prior knowledge of RAG assumed.

---

## Table of Contents

1. [What this system does](#1-what-this-system-does)
2. [Architecture overview](#2-architecture-overview)
3. [Folder structure](#3-folder-structure)
4. [How to run locally](#4-how-to-run-locally)
5. [Key configuration (`.env`)](#5-key-configuration-env)
6. [Common tasks](#6-common-tasks)
7. [How the RAG pipeline works](#7-how-the-rag-pipeline-works)
8. [How document ingestion works](#8-how-document-ingestion-works)
9. [How the chunker works](#9-how-the-chunker-works)
10. [Tuning search quality](#10-tuning-search-quality)
11. [Adding a new LLM or embedding model](#11-adding-a-new-llm-or-embedding-model)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What this system does

Users upload legal PDF documents. The system:

1. Extracts text from the PDF
2. Splits it into overlapping chunks (preserving document hierarchy like Article, Chapter, etc.)
3. Converts each chunk to a vector (embedding) using an AI model
4. Stores the chunks in a vector database (Weaviate)
5. When a user asks a question, finds the most relevant chunks and feeds them to an LLM
6. The LLM answers citing only the found chunks — it cannot hallucinate from outside knowledge

**Key design principle**: the LLM never sees the raw documents, only the relevant excerpts found by vector search. This prevents hallucination and makes the system auditable.

---

## 2. Architecture overview

```
User (browser / Streamlit UI)
         │
         ▼
┌─────────────────────┐
│   FastAPI (API)     │  ← HTTP REST gateway
│   port 8000         │
└────────┬────────────┘
         │                    ┌─────────────┐
         ├── search query ──► │   Weaviate  │  vector DB (ANN search)
         │                    └─────────────┘
         │                    ┌─────────────┐
         ├── job status ────► │  PostgreSQL │  metadata + job status
         │                    └─────────────┘
         │                    ┌─────────────┐
         └── upload PDF ────► │    Redis    │  task queue
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   Celery    │  async worker
                              │   Worker    │  (PDF → chunks → embed → store)
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │    Ollama   │  local LLM (embeddings + generation)
                              └─────────────┘
```

**Flow for upload:**
`User uploads PDF → API saves to PG → sends task to Redis → Worker picks it up → PDF → chunks → embed → Weaviate + PG`

**Flow for search:**
`User asks question → API embeds query → ANN search in Weaviate → rerank results → LLM generates answer`

---

## 3. Folder structure

```
hierarchical-legal-rag/
├── docker-compose.yml          ← start everything with one command
├── .env                        ← your local secrets (never commit this)
├── .env.example                ← copy this to .env and fill in values
├── ui.py                       ← Streamlit demo UI (run standalone)
├── docs/
│   └── DEVELOPER_GUIDE.md      ← this file
│
├── shared/                     ← shared Python package (used by API + worker)
│   └── legal_rag_shared/
│       ├── domain/
│       │   ├── entities.py     ← data models (Document, Chunk, RetrievedChunk)
│       │   └── ports.py        ← abstract interfaces (VectorStore, LLMClient, etc.)
│       ├── db/
│       │   ├── models.py       ← SQLAlchemy ORM models
│       │   └── base.py         ← SQLAlchemy base class
│       └── schemas/
│           └── tasks.py        ← Pydantic schemas for Celery task messages
│
├── services/
│   ├── api/                    ← FastAPI service
│   │   └── app/
│   │       ├── main.py         ← app factory + startup/shutdown hooks
│   │       ├── core/
│   │       │   └── config.py   ← all settings (read from .env)
│   │       ├── api/
│   │       │   ├── deps.py     ← dependency injection (what gets injected where)
│   │       │   ├── middleware.py← request logging + security headers
│   │       │   └── routes/
│   │       │       ├── documents.py  ← POST /v1/documents/upload
│   │       │       ├── jobs.py       ← GET /v1/jobs/{task_id}
│   │       │       └── search.py     ← POST /v1/search/
│   │       ├── application/
│   │       │   ├── rag_service.py    ← RAG pipeline orchestrator ⭐
│   │       │   ├── reranking.py      ← BM25 + cross-encoder reranker ⭐
│   │       │   ├── context_formatting.py ← formats chunks for LLM prompt
│   │       │   ├── query_sanitizer.py    ← blocks prompt injection attacks
│   │       │   ├── retrieval_service.py  ← embed query + ANN search
│   │       │   └── evaluation.py         ← lightweight relevance metrics
│   │       └── infrastructure/
│   │           ├── vector_store_weaviate.py ← Weaviate client (search)
│   │           ├── llm_ollama.py            ← Ollama LLM client
│   │           ├── embedding_ollama.py      ← Ollama embedder (async)
│   │           └── repositories_pg.py       ← PostgreSQL queries
│   │
│   └── worker/                 ← Celery worker service
│       └── worker/
│           ├── celery_app.py   ← Celery app factory
│           ├── tasks/
│           │   └── ingestion.py ← the Celery task (entry point)
│           ├── application/
│           │   ├── ingestion_service.py     ← pipeline orchestrator ⭐
│           │   └── chunking/
│           │       └── legal_chunker.py     ← text splitter ⭐
│           ├── core/
│           │   └── config.py   ← worker settings
│           └── infrastructure/
│               ├── vector_store_weaviate.py ← Weaviate client (write)
│               ├── embedding_ollama.py      ← Ollama embedder (sync)
│               ├── pdf_parser.py            ← PDF text extraction
│               └── repositories_pg.py       ← PostgreSQL queries
│
└── infra/
    ├── postgres/
    │   └── 01_init.sql         ← DB roles and permissions
    └── migrations/
        └── versions/
            └── 0001_initial_schema.py  ← DB table definitions
```

Files marked ⭐ are the ones you'll modify most often.

---

## 4. How to run locally

**Prerequisites:**
- Docker Desktop installed and running
- Ollama installed (`brew install ollama` on Mac)

**Step 1: Start Ollama and pull models**
```bash
ollama serve                          # starts Ollama in background
ollama pull nomic-embed-text          # embedding model (~274 MB)
ollama pull mistral                   # LLM for answers (~4 GB)
```

**Step 2: Configure environment**
```bash
cp .env.example .env
# Edit .env if needed — defaults work for local development
```

**Step 3: Start all services**
```bash
docker-compose up --build
```

This starts: PostgreSQL, Redis, Weaviate, API (port 8000), Worker, Flower (Celery dashboard).

**Step 4: Open the UI**
```bash
pip install streamlit
streamlit run ui.py
```

Open http://localhost:8501

**Step 5: Check everything works**
- API docs: http://localhost:8000/docs
- Flower (worker dashboard): http://localhost:5555
- Health check: http://localhost:8000/health

---

## 5. Key configuration (`.env`)

| Variable | What it does | Default |
|----------|-------------|---------|
| `POSTGRES_*` | Database connection | see .env.example |
| `REDIS_PASSWORD` | Redis auth | see .env.example |
| `OLLAMA_BASE_URL` | Where Ollama runs | `http://host.docker.internal:11434` |
| `OLLAMA_LLM_MODEL` | LLM for answers | `mistral` |
| `OLLAMA_EMBED_MODEL` | Embedding model | `nomic-embed-text` |
| `RAG_RETRIEVAL_TOP_K_DEFAULT` | Chunks returned per query | `5` |
| `RAG_ENABLE_RERANKING` | Use BM25 reranker | `true` |
| `CROSS_ENCODER_MODEL` | Enable cross-encoder reranker | *(empty = disabled)* |
| `RAG_MAX_CHUNK_CHARS` | Max chars per chunk | `1400` |
| `RAG_CHUNK_OVERLAP_CHARS` | Overlap between chunks | `120` |

**To enable the cross-encoder reranker** (better quality, slower):
```env
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```
Also run: `pip install sentence-transformers` in the API container.

---

## 6. Common tasks

### Change the LLM model
Edit `.env`:
```env
OLLAMA_LLM_MODEL=llama3.1
```
Make sure to pull the model first: `ollama pull llama3.1`

### Change the system prompt (how the LLM behaves)
Edit [services/api/app/application/rag_service.py](../services/api/app/application/rag_service.py), the `_SYSTEM_PROMPT` constant at the top of the file.

### Change chunk size
Edit `.env`:
```env
RAG_MAX_CHUNK_CHARS=1000   # smaller = more chunks, more specific
RAG_CHUNK_OVERLAP_CHARS=100
```
**Important**: after changing chunk size, re-upload all documents so they get re-chunked.

### Add support for a new document language
Edit [services/worker/worker/application/chunking/legal_chunker.py](../services/worker/worker/application/chunking/legal_chunker.py).

Add a new regex to `_HEADING_PATTERNS`. Each pattern is a heading level (lower index = higher in hierarchy). Example for Spanish:
```python
re.compile(
    r"^\s*(ARTÍCULO|ART\.?)\s*\d+",
    re.IGNORECASE | re.MULTILINE,
),
```

### Change how many results are returned
Either:
- Set `RAG_RETRIEVAL_TOP_K_DEFAULT` in `.env` (changes the default)
- Or pass `top_k` in the API request body: `{"query": "...", "top_k": 10}`

### Tune the reranker
Edit [services/api/app/application/reranking.py](../services/api/app/application/reranking.py):

```python
_ALPHA = 0.70        # 70% dense (vector) + 30% BM25. Increase for more semantic matching.
_DEDUP_THRESHOLD = 0.70  # Jaccard threshold to drop near-duplicates. Lower = more aggressive dedup.
_K1 = 1.5           # BM25 term saturation
_B = 0.75           # BM25 length normalization
```

### Add a new API route
1. Create a file in `services/api/app/api/routes/yourroute.py`
2. Register it in `services/api/app/main.py`:
   ```python
   from app.api.routes import yourroute
   app.include_router(yourroute.router, prefix="/v1")
   ```

---

## 7. How the RAG pipeline works

When a user asks a question, this happens in `rag_service.py`:

```
User query
    │
    ▼
1. SANITIZE — block prompt injection attacks (query_sanitizer.py)
    │
    ▼
2. RETRIEVE — embed query → ANN search in Weaviate → top 15 candidate chunks
    │          (retrieval_service.py)
    ▼
3. RERANK — score each chunk for relevance and remove near-duplicates
    │        BM25 hybrid (default) or cross-encoder (if configured)
    │        (reranking.py)
    ▼
4. FILTER — drop chunks with score < 0.25 (low relevance)
    │
    ▼
5. FORMAT — number chunks [1], [2]... and build the LLM prompt
    │        (context_formatting.py)
    ▼
6. GENERATE — send prompt to Ollama LLM, get answer
    │          (llm_ollama.py via /api/chat)
    ▼
7. EVALUATE — compute metrics (chunk count, score stats)
    │
    ▼
Answer + sources + metrics → HTTP response
```

**Anti-hallucination design:**
- LLM is instructed to ONLY use the numbered excerpts
- LLM must cite `[1]`, `[2]` for every claim
- If no relevant chunks found, returns a fixed "no information" message
- `temperature=0.0` makes output deterministic and conservative

---

## 8. How document ingestion works

When a PDF is uploaded (`POST /v1/documents/upload`):

```
PDF file
    │
    ▼
API saves DocumentRecord to PostgreSQL (status=PENDING)
    │
    ▼
API sends task_id to Redis queue (THEN commits DB — important order!)
    │
    ▼
Celery Worker picks up the task
    │
    ▼
1. Load document text from PG, mark job as STARTED
    │
    ▼
2. Delete existing chunks (versioning — safe to re-upload a document)
    │  - Deletes from Weaviate first
    │  - Then deletes from PostgreSQL
    ▼
3. Split text into chunks using legal_chunker.py
    │
    ▼
4. Embed all chunks (batch of 32 via Ollama /api/embed)
    │
    ▼
5. Upsert chunks + vectors to Weaviate (batches of 100)
    │
    ▼
6. Save ChunkRecord rows to PostgreSQL
    │
    ▼
Mark job SUCCESS, document status=COMPLETED
```

**Document versioning:** if you upload the same document twice, the old chunks are automatically deleted before the new ones are created. No duplicates.

---

## 9. How the chunker works

The chunker (`legal_chunker.py`) uses regex to detect headings in legal documents and splits text at those boundaries, preserving the document hierarchy.

**Example input:**
```
CHAPITRE II — Sanctions

Article 5
The fine shall not exceed...

Article 6
In case of recidivism...
```

**Example output:**
- Chunk 1: `text="CHAPITRE II...\nArticle 5\nThe fine..."`, `hierarchy_path=["CHAPITRE II — Sanctions", "Article 5"]`
- Chunk 2: `text="Article 6\nIn case of recidivism..."`, `hierarchy_path=["CHAPITRE II — Sanctions", "Article 6"]`

The `hierarchy_path` is stored in Weaviate and shown in the UI as "breadcrumbs" so users can see where in the document a passage comes from.

**Supported languages:** French, Portuguese/Brazilian, generic numbered sections.

**To add a new heading type:** add a regex to `_HEADING_PATTERNS` list in `legal_chunker.py`. The position in the list determines the heading level (0 = document-level, higher = more specific).

---

## 10. Tuning search quality

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Results from wrong topic | Score threshold too low | Increase `_MIN_SCORE` in `rag_service.py` |
| Missing relevant chunks | Score threshold too high | Decrease `_MIN_SCORE` |
| Repeated content in answer | Dedup threshold too high | Decrease `_DEDUP_THRESHOLD` in `reranking.py` |
| Poor exact-match results | BM25 weight too low | Increase `1 - _ALPHA` (decrease `_ALPHA`) |
| Chunks too small/fragmented | `MIN_CHUNK_CHARS` too high | Decrease `_MIN_CHUNK_CHARS` in `legal_chunker.py` |
| Chunks too large | `max_chunk_chars` too big | Decrease `RAG_MAX_CHUNK_CHARS` in `.env` |
| "No hierarchy" in results | Heading not detected | Add regex pattern to `_HEADING_PATTERNS` |
| LLM hallucinating | Chunks below threshold reaching LLM | Increase `_MIN_SCORE` |

**Best quality reranking:** set `CROSS_ENCODER_MODEL=BAAI/bge-reranker-base` in `.env`. This is significantly better than BM25, especially for French/Portuguese documents. Requires `pip install sentence-transformers`.

---

## 11. Adding a new LLM or embedding model

The system uses **ports** (interfaces) for LLM and embedding. Any implementation that matches the interface works.

**To add OpenAI GPT as LLM:**
1. Create `services/api/app/infrastructure/llm_openai.py`
2. Implement the `async def generate(self, prompt: str, *, system: str = "") -> str` method
3. In `services/api/app/main.py`, replace `AsyncOllamaLLMClient` with your new class
4. Add `OPENAI_API_KEY` to `.env`

The key method signatures are in `shared/legal_rag_shared/domain/ports.py`.

**To change embedding model:** just update `OLLAMA_EMBED_MODEL` in `.env` and re-upload documents (the new model will produce different vectors, old ones will be wrong).

---

## 12. Troubleshooting

### "No chunks above score threshold"
The document might not have been ingested yet, or the query doesn't match the document language. Check:
1. Job status: `GET /v1/jobs/{task_id}`
2. Is Ollama running? `curl http://localhost:11434/api/tags`
3. Is Weaviate running? `curl http://localhost:8080/v1/meta`

### Worker not processing tasks
Check Flower at http://localhost:5555. Also check worker logs:
```bash
docker-compose logs worker
```

### "Fragment" chunks in results (very short, incomplete sentences)
The chunker's `_MIN_CHUNK_CHARS` (default 80) filters short chunks. If you're still seeing fragments, the overlap may be creating them. Reduce `RAG_CHUNK_OVERLAP_CHARS` in `.env`.

### Document re-uploaded but old results still appear
This is a cache issue. The versioning system deletes old chunks, but Weaviate's HNSW index has eventual consistency. Wait a few seconds and retry.

### Ollama batch embedding fails with "unexpected shape"
Your Ollama version doesn't support the batch `/api/embed` endpoint. Upgrade Ollama:
```bash
brew upgrade ollama   # macOS
```
Or revert to sequential embedding by changing `embed_texts` in `embedding_ollama.py` to call `embed_text` in a loop (temporary workaround).

### How to reset everything (clean slate)
```bash
docker-compose down -v   # removes all volumes (DELETES ALL DATA)
docker-compose up --build
```

---

## Key files quick reference

| What you want to change | File |
|------------------------|------|
| LLM answer behavior / system prompt | `services/api/app/application/rag_service.py` |
| Search ranking algorithm | `services/api/app/application/reranking.py` |
| Document chunking / heading detection | `services/worker/worker/application/chunking/legal_chunker.py` |
| API configuration / env vars | `services/api/app/core/config.py` |
| Worker configuration / env vars | `services/worker/worker/core/config.py` |
| API routes | `services/api/app/api/routes/` |
| Database table definitions | `infra/migrations/versions/0001_initial_schema.py` |
| Docker services | `docker-compose.yml` |
| Dependency injection wiring | `services/api/app/api/deps.py` |
