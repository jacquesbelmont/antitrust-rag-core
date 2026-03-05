"""
API service configuration — all values sourced from environment variables.

Pydantic-settings reads from environment variables (and optionally a .env
file).  Every field without a default MUST be present in the environment;
the application will refuse to start with a clear error message if not.

Never add default values for secrets (passwords, keys).  Use `...` (required)
so that a misconfigured deployment fails loudly at startup rather than
silently connecting to a wrong or open database.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Extra env vars from Docker / the host are silently ignored.
        extra="ignore",
    )

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    postgres_host: str = Field(..., description="PostgreSQL hostname")
    postgres_port: int = Field(5432, description="PostgreSQL port")
    postgres_db: str = Field(..., description="PostgreSQL database name")
    postgres_user: str = Field(..., description="PostgreSQL username")
    postgres_password: str = Field(..., description="PostgreSQL password")
    # Emit SQL to stdout — useful for development, must be False in production.
    db_echo: bool = Field(False, description="Echo SQL statements (dev only)")

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_host: str = Field(..., description="Redis hostname")
    redis_port: int = Field(6379, description="Redis port")
    redis_password: str = Field(..., description="Redis password")

    # ── Weaviate ───────────────────────────────────────────────────────────────
    weaviate_url: str = Field(
        "http://weaviate:8080", description="Weaviate HTTP endpoint"
    )

    # ── Ollama ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        "http://host.docker.internal:11434", description="Ollama base URL"
    )
    ollama_llm_model: str = Field("mistral", description="LLM model name")
    ollama_embed_model: str = Field(
        "nomic-embed-text", description="Embedding model name"
    )

    # ── RAG pipeline ───────────────────────────────────────────────────────────
    rag_retrieval_top_k_default: int = Field(5, ge=1, le=50)
    rag_enable_reranking: bool = Field(True)
    # Cross-encoder reranker (optional — requires sentence-transformers).
    # Leave empty to use the BM25 hybrid reranker (default).
    # Example values:
    #   cross-encoder/ms-marco-MiniLM-L-6-v2  (~80 MB, fast, English)
    #   BAAI/bge-reranker-base                 (~270 MB, multilingual FR/PT/EN)
    cross_encoder_model: str = Field(
        "",
        description="HuggingFace cross-encoder model name. Empty = use BM25 hybrid.",
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")

    @property
    def celery_broker_url(self) -> str:
        return (
            f"redis://:{self.redis_password}"
            f"@{self.redis_host}:{self.redis_port}/0"
        )

    @property
    def celery_result_backend(self) -> str:
        return (
            f"redis://:{self.redis_password}"
            f"@{self.redis_host}:{self.redis_port}/1"
        )
