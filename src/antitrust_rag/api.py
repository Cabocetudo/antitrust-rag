"""FastAPI application: /health, /ingest, /search, /ask (OpenAPI docs at /docs)."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from antitrust_rag import __version__
from antitrust_rag.config import Settings, get_settings
from antitrust_rag.generation import LLMError
from antitrust_rag.logging_setup import configure_logging
from antitrust_rag.models import ScoredChunk
from antitrust_rag.service import RAGService, build_service
from antitrust_rag.sources import SourceUnavailableError, load_sample_dir

logger = logging.getLogger("antitrust_rag.api")

SourceId = Literal["lei-12529-2011", "tfeu-101", "tfeu-102"]


# --- schemas ---------------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000, examples=["What counts as a cartel?"])
    top_k: int = Field(default=6, ge=1, le=20)
    mode: Literal["hybrid", "lexical", "vector"] = "hybrid"
    sources: list[SourceId] | None = Field(default=None, description="Restrict to these sources")
    rerank: bool | None = Field(default=None, description="Override the server rerank default")


class ChunkOut(BaseModel):
    chunk_id: str
    citation: str
    source_id: str
    url: str | None
    text: str
    score: float
    ranks: dict[str, int]

    @classmethod
    def from_scored(cls, sc: ScoredChunk) -> ChunkOut:
        c = sc.chunk
        return cls(
            chunk_id=c.chunk_id,
            citation=c.citation,
            source_id=c.source_id,
            url=c.url,
            text=c.text,
            score=round(sc.score, 6),
            ranks=sc.ranks,
        )


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[ChunkOut]
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["Which conducts are considered infringements of the economic order in Brazil?"],
    )
    top_k: int = Field(default=6, ge=1, le=20)
    sources: list[SourceId] | None = None
    rerank: bool | None = None
    use_cache: bool = True


class CitationOut(BaseModel):
    citation: str
    supported: bool = Field(description="True if it matches a source sent to the model")


class UsageOut(BaseModel):
    model: str | None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    llm_attempts: int


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: list[CitationOut]
    sources: list[ChunkOut]
    usage: UsageOut
    cached: bool
    timings_ms: dict[str, float]


class IngestRequest(BaseModel):
    origin: Literal["sample", "remote"] = Field(
        default="sample",
        description="'sample' = bundled offline excerpts; 'remote' = download official texts",
    )
    sources: list[SourceId] | None = None


class IngestResponse(BaseModel):
    origin: str
    chunks_written: dict[str, int]
    total_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    store_ok: bool
    chunks_indexed: int
    llm_configured: bool
    embedder: str
    metrics: dict[str, float | int]


def _ids(sources: list[SourceId] | None) -> list[str] | None:
    return [str(s) for s in sources] if sources else None


def _service(request: Request) -> RAGService:
    return request.app.state.service


Service = Annotated[RAGService, Depends(_service)]


# --- app factory -----------------------------------------------------------------------------
def create_app(service: RAGService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if getattr(app.state, "service", None) is None:
            app.state.service = build_service(settings)
        yield

    app = FastAPI(
        title="Antitrust RAG",
        version=__version__,
        description=(
            "Question answering over public competition-law texts (Brazil Lei 12.529/2011, "
            "TFEU Arts. 101-102) with hybrid retrieval and cited answers. "
            "Legal information, not legal advice."
        ),
        lifespan=lifespan,
    )
    app.state.service = service

    @app.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        logger.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(s: Service) -> HealthResponse:
        store_ok = s.store.ping()
        m = s.metrics
        return HealthResponse(
            status="ok" if store_ok else "degraded",
            version=__version__,
            store_ok=store_ok,
            chunks_indexed=s.store.count() if store_ok else 0,
            llm_configured=s.llm is not None,
            embedder=s.embedder.name,
            metrics={
                "asks": m.asks,
                "searches": m.searches,
                "llm_calls": m.llm_calls,
                "llm_errors": m.llm_errors,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "estimated_cost_usd": round(m.cost_usd, 6),
                "cache_hits": s.cache.hits,
                "cache_misses": s.cache.misses,
            },
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest, s: Service) -> IngestResponse:
        try:
            if req.origin == "sample":
                docs = [
                    d
                    for d in load_sample_dir(Path(settings.sample_dir))
                    if not req.sources or d.spec.source_id in req.sources
                ]
                written = await run_in_threadpool(s.ingest_documents, docs)
            else:
                written = await run_in_threadpool(s.ingest_remote, _ids(req.sources))
        except SourceUnavailableError as exc:
            raise HTTPException(status_code=502, detail=f"source unavailable: {exc}") from exc
        return IngestResponse(
            origin=req.origin, chunks_written=written, total_chunks=s.store.count()
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest, s: Service) -> SearchResponse:
        res = await run_in_threadpool(
            lambda: s.search(
                req.query, req.top_k, mode=req.mode, sources=_ids(req.sources), rerank=req.rerank
            )
        )
        return SearchResponse(
            query=req.query,
            mode=res.mode,
            results=[ChunkOut.from_scored(c) for c in res.chunks],
            timings_ms={k: round(v, 2) for k, v in res.timings_ms.items()},
        )

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, s: Service) -> AskResponse:
        try:
            res = await run_in_threadpool(
                lambda: s.ask(
                    req.question,
                    req.top_k,
                    sources=_ids(req.sources),
                    rerank=req.rerank,
                    use_cache=req.use_cache,
                )
            )
        except LLMError as exc:
            logger.error("ask failed", extra={"error": str(exc)})
            raise HTTPException(
                status_code=503,
                detail="Answer generation is temporarily unavailable. "
                "Retrieval still works: try POST /search.",
            ) from exc
        a = res.answer
        return AskResponse(
            question=req.question,
            answer=a.answer,
            grounded=a.grounded,
            citations=[CitationOut(citation=c.text, supported=c.supported) for c in a.citations],
            sources=[ChunkOut.from_scored(c) for c in a.sources],
            usage=UsageOut(
                model=a.usage.model if a.usage else None,
                input_tokens=a.usage.input_tokens if a.usage else 0,
                output_tokens=a.usage.output_tokens if a.usage else 0,
                estimated_cost_usd=a.cost_usd,
                llm_attempts=a.usage.attempts if a.usage else 0,
            ),
            cached=res.cached,
            timings_ms={k: round(v, 2) for k, v in res.timings_ms.items()},
        )

    return app
