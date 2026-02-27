from __future__ import annotations

from functools import lru_cache

from app.application.context_formatting import ContextFormatter
from app.application.evaluation import EvaluationService
from app.application.ingestion_service import IngestionService
from app.application.rag_service import RAGService
from app.application.reranking import BM25Reranker
from app.application.retrieval_service import RetrievalService
from app.core.config import settings
from app.infrastructure.clock import UtcClock
from app.infrastructure.embedding import DeterministicHashEmbedder
from app.infrastructure.id_generator import Uuid7LikeIdGenerator
from app.infrastructure.llm_mock import MockLLMClient
from app.infrastructure.repositories import InMemoryChunkRepository, InMemoryDocumentRepository
from app.infrastructure.vector_store_in_memory import InMemoryVectorStore


@lru_cache(maxsize=1)
def container() -> dict:
    # --- Embedder ---
    # Switch RAG_USE_OLLAMA=true to use local Ollama embeddings (nomic-embed-text).
    # Falls back to the deterministic hash embedder automatically when Ollama is unavailable.
    if settings.use_ollama:
        from app.infrastructure.embedding_ollama import OllamaEmbedder
        embedder = OllamaEmbedder()
    else:
        embedder = DeterministicHashEmbedder()

    # --- Vector store ---
    # Switch RAG_VECTOR_STORE_BACKEND=weaviate to use Weaviate (requires `docker compose up weaviate`).
    if settings.vector_store_backend == "weaviate":
        from app.infrastructure.vector_store_weaviate import WeaviateVectorStore
        vector_store = WeaviateVectorStore()
    else:
        vector_store = InMemoryVectorStore()

    # --- LLM ---
    # Switch RAG_USE_OLLAMA=true to use a local Mistral/Llama model via Ollama.
    if settings.use_ollama:
        from app.infrastructure.llm_ollama import OllamaLLMClient
        llm = OllamaLLMClient()
    else:
        llm = MockLLMClient()

    # --- Reranker ---
    reranker: BM25Reranker | None = BM25Reranker() if settings.enable_reranking else None

    retrieval = RetrievalService(vector_store=vector_store, embedder=embedder)
    formatter = ContextFormatter()
    evaluation = EvaluationService()

    rag = RAGService(
        retrieval=retrieval,
        formatter=formatter,
        llm=llm,
        evaluation=evaluation,
        reranker=reranker,
    )

    docs_repo = InMemoryDocumentRepository()
    chunks_repo = InMemoryChunkRepository()

    ingestion = IngestionService(
        documents=docs_repo,
        chunks_repo=chunks_repo,
        vector_store=vector_store,
        embedder=embedder,
        ids=Uuid7LikeIdGenerator(),
        clock=UtcClock(),
    )

    return {
        "rag": rag,
        "ingestion": ingestion,
    }


def get_rag_service() -> RAGService:
    return container()["rag"]


def get_ingestion_service() -> IngestionService:
    return container()["ingestion"]
