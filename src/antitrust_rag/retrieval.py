"""Hybrid retrieval: lexical + dense candidates fused with Reciprocal Rank Fusion, optional rerank.

Why RRF? Lexical scores (BM25 / ts_rank_cd) and cosine similarities live on unrelated scales, so
adding them needs per-query normalization that is fragile. RRF only uses *ranks*:

    rrf(d) = Σ_r  1 / (k + rank_r(d))

It is parameter-light (k=60 from Cormack et al., 2009), robust, and cheap. Legal questions need
both signals: exact terms ("leniência", "Article 102", "20%") favour lexical; paraphrases and
cross-language questions ("price-fixing cartel" vs "acordar preços com concorrente") favour dense.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from antitrust_rag.embeddings import Embedder
from antitrust_rag.models import ScoredChunk
from antitrust_rag.store.base import VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ScoredChunk]], k: int = 60, top_n: int | None = None
) -> list[ScoredChunk]:
    """Fuse several ranked lists by chunk id. Keeps per-list ranks on the result for debugging."""
    fused: dict[str, ScoredChunk] = {}
    for name, results in ranked_lists.items():
        for rank, sc in enumerate(results, start=1):
            cid = sc.chunk.chunk_id
            entry = fused.setdefault(cid, ScoredChunk(sc.chunk, 0.0))
            entry.score += 1.0 / (k + rank)
            entry.ranks[name] = rank
    out = sorted(fused.values(), key=lambda s: (-s.score, s.chunk.chunk_id))
    return out[:top_n] if top_n else out


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Cross-encoder reranker (multilingual MiniLM trained on mMARCO by default).

    Scores (query, passage) pairs jointly — more accurate than bi-encoder similarity but
    O(candidates) model calls per query, so it runs only on the fused top-N.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, keep it lazy

        self._model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        for pos, (c, s) in enumerate(zip(candidates, scores, strict=True), start=1):
            c.ranks["fused"] = pos
            c.score = float(s)
        return sorted(candidates, key=lambda c: -c.score)[:top_k]


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    mode: str = "hybrid"


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        candidate_k: int = 30,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        rerank_pool: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_pool = rerank_pool

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        mode: str = "hybrid",
        sources: list[str] | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        lists: dict[str, list[ScoredChunk]] = {}

        if mode in ("hybrid", "lexical"):
            t0 = time.perf_counter()
            lists["lexical"] = self.store.lexical_search(query, self.candidate_k, sources)
            timings["lexical_ms"] = (time.perf_counter() - t0) * 1000
        if mode in ("hybrid", "vector"):
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            lists["vector"] = self.store.vector_search(qvec, self.candidate_k, sources)
            timings["vector_ms"] = (time.perf_counter() - t0) * 1000
        if not lists:
            raise ValueError(f"unknown retrieval mode {mode!r}")

        if len(lists) == 1:
            ((name, results),) = lists.items()
            for i, r in enumerate(results, start=1):
                r.ranks[name] = i
            fused = results
        else:
            fused = reciprocal_rank_fusion(lists, k=self.rrf_k)

        use_rerank = self.reranker is not None if rerank is None else rerank
        if use_rerank and self.reranker is not None:
            t0 = time.perf_counter()
            chunks = self.reranker.rerank(query, fused[: self.rerank_pool], top_k)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        else:
            if use_rerank:
                logger.warning("rerank requested but no reranker configured; skipping")
            chunks = fused[:top_k]
        return RetrievalResult(chunks=chunks, timings_ms=timings, mode=mode)
