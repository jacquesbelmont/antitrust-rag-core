"""
Worker service configuration — all values sourced from environment variables.

Mirrors the API's Settings class for shared datastore access, but omits
API-only concerns (LLM model, top-k) and adds worker-specific tuning.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    postgres_host: str = Field(..., description="PostgreSQL hostname")
    postgres_port: int = Field(5432)
    postgres_db: str = Field(..., description="PostgreSQL database name")
    postgres_user: str = Field(..., description="PostgreSQL username")
    postgres_password: str = Field(..., description="PostgreSQL password")
    db_echo: bool = Field(False)

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_host: str = Field(..., description="Redis hostname")
    redis_port: int = Field(6379)
    redis_password: str = Field(..., description="Redis password")

    # ── Weaviate ───────────────────────────────────────────────────────────────
    weaviate_url: str = Field("http://weaviate:8080")

    # ── Ollama (embedding only in the worker) ──────────────────────────────────
    ollama_base_url: str = Field("http://host.docker.internal:11434")
    ollama_embed_model: str = Field("nomic-embed-text")

    # ── Chunking pipeline ──────────────────────────────────────────────────────
    rag_max_chunk_chars: int = Field(1400, ge=200, le=8000)
    rag_chunk_overlap_chars: int = Field(120, ge=0, le=500)

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
