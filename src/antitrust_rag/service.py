"""Application service: wires ingestion, retrieval, generation, caching and metrics together.

The FastAPI layer, the CLI and the evaluation script all go through this class, so there is one
code path for "how a question is answered".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from antitrust_rag.cache import TTLCache
from antitrust_rag.chunking import chunk_document
from antitrust_rag.config import Settings
from antitrust_rag.embeddings import Embedder, build_embedder
from antitrust_rag.generation import (
    NOT_FOUND,
    SYSTEM_PROMPT,
    Answer,
    ClaudeClient,
    LLMClient,
    LLMError,
    build_context,
    build_user_prompt,
    check_citations,
    estimate_cost_usd,
    is_not_found,
)
from antitrust_rag.retrieval import CrossEncoderReranker, HybridRetriever, RetrievalResult
from antitrust_rag.sources import SOURCES, RawDocument, fetch_source, load_sample_dir
from antitrust_rag.store.base import VectorStore
from antitrust_rag.store.memory import InMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class AskResult:
    answer: Answer
    cached: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Metrics:
    """Process-local counters exposed on /health (a Prometheus exporter is on the roadmap)."""

    asks: int = 0
    searches: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class RAGService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.retriever = HybridRetriever(
            store, embedder, candidate_k=s.candidate_k, rrf_k=s.rrf_k, reranker=reranker
        )
        self.cache = TTLCache(max_items=s.cache_max_items, ttl_s=s.cache_ttl_s)
        self.metrics = Metrics()
        self.index_version = 0
        self._lock = threading.Lock()

    # -- ingestion ---------------------------------------------------------------------------
    def ingest_documents(self, docs: list[RawDocument], batch_size: int = 64) -> dict[str, int]:
        written: dict[str, int] = {}
        for doc in docs:
            t0 = time.perf_counter()
            chunks = chunk_document(doc)
            self.store.delete_source(doc.spec.source_id)  # re-ingest replaces, never duplicates
            n = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                n += self.store.upsert(
                    batch, self.embedder.embed_documents([c.text for c in batch])
                )
            written[doc.spec.source_id] = n
            logger.info(
                "ingested source",
                extra={
                    "source": doc.spec.source_id,
                    "chunks": n,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                },
            )
        with self._lock:
            self.index_version += 1
        self.cache.clear()
        return written

    def ingest_sample(self, directory: Path) -> dict[str, int]:
        return self.ingest_documents(load_sample_dir(directory))

    def ingest_remote(self, source_ids: list[str] | None = None) -> dict[str, int]:
        ids = source_ids or list(SOURCES)
        docs = []
        for sid in ids:
            spec = SOURCES[sid]
            url, lines = fetch_source(spec)
            docs.append(RawDocument(spec=spec, url=url, lines=lines))
        return self.ingest_documents(docs)

    # -- retrieval ---------------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        mode: str = "hybrid",
        sources: list[str] | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        self.metrics.searches += 1
        return self.retriever.retrieve(
            query, top_k or self.settings.top_k, mode=mode, sources=sources, rerank=rerank
        )

    # -- generation --------------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        *,
        sources: list[str] | None = None,
        rerank: bool | None = None,
        use_cache: bool = True,
    ) -> AskResult:
        s = self.settings
        self.metrics.asks += 1
        top_k = top_k or s.top_k
        key = TTLCache.make_key(
            q=" ".join(question.lower().split()),
            k=top_k,
            sources=sorted(sources or []),
            rerank=rerank,
            model=getattr(self.llm, "model", None),
            v=self.index_version,
        )
        if use_cache and (hit := self.cache.get(key)) is not None:
            return AskResult(answer=hit, cached=True, timings_ms={})

        t_total = time.perf_counter()
        retrieval = self.search(question, top_k, sources=sources, rerank=rerank)
        timings = dict(retrieval.timings_ms)

        if not retrieval.chunks:
            # Nothing retrieved: skip the LLM call entirely (zero cost, no hallucination risk).
            answer = Answer(answer=NOT_FOUND, grounded=False)
        else:
            if self.llm is None:
                raise LLMError("generation is disabled: set ANTHROPIC_API_KEY")
            context, used = build_context(retrieval.chunks, s.max_context_chars)
            try:
                resp = self.llm.complete(
                    SYSTEM_PROMPT, build_user_prompt(question, context), s.llm_max_tokens
                )
            except LLMError:
                self.metrics.llm_errors += 1
                raise
            self.metrics.llm_calls += 1
            self.metrics.input_tokens += resp.input_tokens
            self.metrics.output_tokens += resp.output_tokens
            cost = estimate_cost_usd(
                resp.model,
                resp.input_tokens,
                resp.output_tokens,
                (s.llm_price_input_per_mtok, s.llm_price_output_per_mtok),
            )
            if cost is not None:
                self.metrics.cost_usd += cost
            timings["llm_ms"] = resp.latency_ms
            not_found = is_not_found(resp.text)
            answer = Answer(
                answer=resp.text,
                grounded=not not_found,
                citations=[] if not_found else check_citations(resp.text, used),
                sources=used,
                usage=resp,
                cost_usd=cost,
            )
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        self.cache.set(key, answer)
        logger.info(
            "ask",
            extra={
                "grounded": answer.grounded,
                "n_sources": len(answer.sources),
                "input_tokens": answer.usage.input_tokens if answer.usage else 0,
                "output_tokens": answer.usage.output_tokens if answer.usage else 0,
                "cost_usd": answer.cost_usd,
                **{k: round(v, 1) for k, v in timings.items()},
            },
        )
        return AskResult(answer=answer, cached=False, timings_ms=timings)


def build_service(settings: Settings) -> RAGService:
    """Construct the service from settings (real model, real DB, real Claude if a key is set)."""
    embedder = build_embedder(settings.embedder, settings.embedding_model, settings.embedding_dim)
    if embedder.dim != settings.embedding_dim:
        raise ValueError(
            f"embedder dim {embedder.dim} != RAG_EMBEDDING_DIM {settings.embedding_dim}"
        )
    store: VectorStore
    if settings.store == "postgres":
        from antitrust_rag.store.postgres import PostgresStore

        pg = PostgresStore(settings.database_url, settings.embedding_dim)
        pg.init_schema()
        store = pg
    else:
        store = InMemoryStore(settings.embedding_dim)

    llm: LLMClient | None = None
    if settings.anthropic_api_key:
        llm = ClaudeClient(
            settings.anthropic_api_key,
            settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    else:
        logger.warning("ANTHROPIC_API_KEY not set: /ask disabled, /search still works")

    reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank else None
    return RAGService(store, embedder, llm=llm, settings=settings, reranker=reranker)
