from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", case_sensitive=False)

    # --- Vector store ---
    # "in_memory" (default, no deps) | "weaviate"
    vector_store_backend: str = "in_memory"

    # --- Chunking ---
    max_chunk_chars: int = 1400
    chunk_overlap_chars: int = 120

    # --- Retrieval ---
    retrieval_top_k_default: int = 5
    enable_reranking: bool = True

    # --- Ollama (local LLM + embeddings, zero token cost) ---
    # Start Ollama: `ollama serve`
    # Pull models: `ollama pull mistral` && `ollama pull nomic-embed-text`
    use_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"            # strong French support
    ollama_embed_model: str = "nomic-embed-text"  # 768-dim, multilingual

    # --- Weaviate ---
    weaviate_url: str = "http://localhost:8080"

    # --- PostgreSQL (uncomment to swap in-memory repos for production) ---
    # postgres_url: str = "postgresql+asyncpg://user:pass@localhost/antitrust_rag"


settings = Settings()
