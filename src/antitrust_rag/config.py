"""Runtime configuration, read from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- storage -----------------------------------------------------------------------------
    store: Literal["postgres", "memory"] = "postgres"
    database_url: str = "postgresql://rag:rag@localhost:5432/antitrust_rag"

    # --- embeddings --------------------------------------------------------------------------
    embedder: Literal["sentence-transformers", "fake"] = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # --- retrieval ---------------------------------------------------------------------------
    top_k: int = Field(default=6, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=1, le=200)  # per-retriever candidates fed into RRF
    rrf_k: int = Field(default=60, ge=1)
    rerank: bool = False
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # --- generation (Anthropic Claude) -------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 800
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    # Optional price override (USD per million tokens). When unset, a built-in table is used.
    llm_price_input_per_mtok: float | None = None
    llm_price_output_per_mtok: float | None = None
    max_context_chars: int = 12_000  # hard cap on retrieved text sent to the model

    sample_dir: str = "data/sample"  # offline corpus bundled with the repo

    # --- API ---------------------------------------------------------------------------------
    cache_ttl_s: int = 600
    cache_max_items: int = 256
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
