"""Storage interface. Two implementations share it:

* :class:`~antitrust_rag.store.postgres.PostgresStore` — production: pgvector HNSW + tsvector/GIN.
* :class:`~antitrust_rag.store.memory.InMemoryStore` — tests and the offline demo: numpy cosine +
  a textbook BM25. Same contract, so retrieval code and the API never know which one is in use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from antitrust_rag.models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """Insert or replace chunks (keyed by ``chunk_id``). Returns number written."""
        ...

    def delete_source(self, source_id: str) -> int:
        """Remove every chunk of a source (used before re-ingesting it). Returns rows deleted."""
        ...

    def vector_search(
        self, query_vec: np.ndarray, k: int, sources: list[str] | None = None
    ) -> list[ScoredChunk]: ...

    def lexical_search(
        self, query: str, k: int, sources: list[str] | None = None
    ) -> list[ScoredChunk]: ...

    def count(self) -> int: ...

    def ping(self) -> bool: ...
