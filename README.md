# Antitrust RAG Core

A production-oriented RAG pipeline for legal documents — built from first principles, without LangChain or LlamaIndex.

## Pipeline

```
ingest(text)
  └─ hierarchical_chunk()       # respects Title > Chapter > Section > Article > Attendu que
       └─ embed(chunks)         # Ollama / nomic-embed-text (local, zero token cost)
            └─ upsert(weaviate) # HNSW, cosine distance

query(text)
  └─ embed(query)
       └─ dense_retrieve(top_k×3)   # Weaviate near_vector
            └─ bm25_rerank(top_k)   # hybrid: 70% semantic + 30% keyword
                 └─ format_context()
                      └─ llm(prompt) # Ollama / Mistral (local) or any LLMClient
                           └─ evaluate(context_relevance, faithfulness_proxy)
```

## Why First Principles?

Frameworks like LangChain abstract away exactly the decisions that matter at scale:
- **Chunking strategy** — a naive character split destroys a legal precedent hierarchy.
- **Reranking** — dense retrieval alone misses exact article citations and case numbers.
- **Evaluation** — you cannot ship to lawyers without measuring correctness, not just plausibility.

This codebase makes every decision explicit and swappable via ports & adapters.

## Architecture

```
app/
├── domain/          # Entities (Document, Chunk, RetrievedChunk) + Ports (ABC/Protocol)
├── application/     # Business logic — no framework, no I/O
│   ├── chunking/    # legal_chunker.py — hierarchical regex chunking (EN/FR/PT/ES)
│   ├── reranking.py # BM25Reranker — hybrid dense + sparse scoring
│   ├── rag_service.py
│   ├── ingestion_service.py
│   ├── retrieval_service.py
│   └── evaluation.py
├── infrastructure/  # Swappable implementations behind Port interfaces
│   ├── embedding.py              # DeterministicHashEmbedder (tests/CI)
│   ├── embedding_ollama.py       # OllamaEmbedder — nomic-embed-text, 768-dim
│   ├── llm_mock.py               # MockLLMClient (tests/CI)
│   ├── llm_ollama.py             # OllamaLLMClient — Mistral / Llama3 (local)
│   ├── vector_store_in_memory.py # InMemoryVectorStore (tests/CI)
│   └── vector_store_weaviate.py  # WeaviateVectorStore — HNSW, production-ready
├── api/             # FastAPI routes — thin, no business logic
└── core/            # Config (pydantic-settings, RAG_* env vars), logging
```

Everything in `application/` is pure Python. Swapping the vector store, embedder, or LLM requires **zero changes** to business logic — only `deps.py` wiring changes.

## Tech Stack

| Layer | Default (local/CI) | Production |
|---|---|---|
| API | FastAPI + uvicorn | Same |
| Vector store | In-memory (cosine) | Weaviate 1.27 (HNSW) |
| Embeddings | Deterministic hash | Ollama `nomic-embed-text` |
| LLM | Mock (echo) | Ollama `mistral` / Claude / GPT-4o |
| Document store | In-memory dict | PostgreSQL (asyncpg) |
| Infra | — | Docker Compose / GCP Cloud Run |

## French Legal Document Chunking

French court decisions and competition authority rulings (Autorité de la concurrence) have specific structural patterns. The chunker detects:

| Pattern | Examples |
|---|---|
| Document-level | `ARRÊT`, `JUGEMENT`, `ORDONNANCE`, `DÉCISION N° 24-D-01` |
| Title | `TITRE I`, `TITLE I` |
| Chapter | `CHAPITRE I`, `CHAPTER I` |
| Section | `SECTION 1`, `§ 3` |
| Article | `ARTICLE L. 420-1`, `Art. 102 TFEU` |
| Paragraph split | `Attendu que`, `Considérant que`, `Vu l'article`, `(1)` |

The hierarchy is tracked as a path (`["ARRÊT du 12 mars 2024", "TITRE I", "SECTION 2"]`) and stored alongside each chunk. Retrieval results include the full path, enabling traceable citations.

## Running Locally

### Option A — In-memory (no dependencies, tests included)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Run tests
pytest
```

### Option B — Full stack (Weaviate + local LLM via Ollama)

```bash
# 1. Start Ollama and pull models (one-time)
ollama serve
ollama pull mistral
ollama pull nomic-embed-text

# 2. Start Weaviate
docker compose up weaviate -d

# 3. Run API with full stack
RAG_VECTOR_STORE_BACKEND=weaviate \
RAG_USE_OLLAMA=true \
RAG_ENABLE_RERANKING=true \
uvicorn app.main:app --reload
```

### Option C — Full Docker (API + Weaviate)

```bash
# Ollama must be running on the host first
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info and endpoint map |
| `GET` | `/health` | Health check |
| `POST` | `/documents/` | Ingest a document (chunk + embed + store) |
| `POST` | `/search/` | RAG query (retrieve + rerank + LLM answer) |
| `GET` | `/docs` | Swagger UI |

### Ingest

```bash
curl -X POST http://localhost:8000/documents/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Décision n° 24-D-01 du 15 janvier 2024",
    "source": "Autorité de la concurrence",
    "text": "SECTION 1 Faits\n\nAttendu que la société X a conclu..."
  }'
```

### Search

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelles sont les pratiques anticoncurrentielles retenues?", "top_k": 5}'
```

Response includes:
- `answer` — LLM response grounded in retrieved context
- `context` — chunks used, with hierarchy paths and scores
- `evaluation` — `context_relevance`, `faithfulness_proxy`, `chunks_retrieved`, `reranking_enabled`

## Configuration (env vars, prefix `RAG_`)

| Variable | Default | Description |
|---|---|---|
| `RAG_VECTOR_STORE_BACKEND` | `in_memory` | `in_memory` or `weaviate` |
| `RAG_USE_OLLAMA` | `false` | Enable local LLM + embeddings |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_OLLAMA_MODEL` | `mistral` | LLM model name |
| `RAG_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_WEAVIATE_URL` | `http://localhost:8080` | Weaviate URL |
| `RAG_ENABLE_RERANKING` | `true` | Toggle BM25 hybrid reranking |
| `RAG_MAX_CHUNK_CHARS` | `1400` | Max chars per chunk |
| `RAG_CHUNK_OVERLAP_CHARS` | `120` | Overlap between size-bounded chunks |
| `RAG_RETRIEVAL_TOP_K_DEFAULT` | `5` | Default top-K for retrieval |

## Reranking Design

The `BM25Reranker` implements hybrid scoring without external ML models:

```
hybrid_score = alpha × dense_score + (1 - alpha) × bm25_score
```

- **Dense (70%)**: semantic similarity from embedding cosine distance.
- **BM25 (30%)**: exact term matching — critical for article numbers, case IDs, company names.
- The system over-retrieves by 3× before reranking, then returns `top_k` results.

For higher precision in production: replace with a cross-encoder (e.g., `ms-marco-MiniLM-L-12-v2`).

## Weaviate Schema

Collection `LegalChunk`, auto-created on first upsert:

- HNSW index, cosine distance
- `ef_construction=128`, `max_connections=64` — good recall/build-time balance for 430K docs
- Filterable properties: `chunk_id`, `document_id`
- For production: add tenant isolation per matter/client

## Evaluation

Every `/search/` response returns:

- **context_relevance**: query token overlap with retrieved context — proxy for retrieval precision.
- **faithfulness_proxy**: answer token overlap with context — proxy for hallucination rate.

These are intentionally simple (no LLM-as-judge, no corpus statistics) to keep the PoC dependency-free. Next step: integrate RAGAS or a domain expert annotation pipeline.

## Production Roadmap

- [ ] PostgreSQL document/chunk repository (swap `InMemoryDocumentRepository`)
- [ ] Batch ingestion endpoint for bulk PDF import (430K docs)
- [ ] PDF parsing layer (pdfminer / PyMuPDF) before chunking
- [ ] IDF computation from corpus for accurate BM25
- [ ] Cross-encoder reranker (sentence-transformers)
- [ ] Caching layer (Redis) for repeated queries
- [ ] GCP Cloud Run deployment + Artifact Registry
- [ ] Pulumi infrastructure-as-code

## Engineering Philosophy

I don't claim to know every framework released last week. What I do know: **data pipelines, embeddings, and reliable infrastructure are not magic** — they are engineering decisions with explicit trade-offs.

Every comment in this codebase explains *why*, not *what*. Every dependency is justified. Every interface has a swap path.
