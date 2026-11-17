from __future__ import annotations

import re
from pathlib import Path

import pytest

from antitrust_rag.config import Settings
from antitrust_rag.embeddings import FakeEmbedder
from antitrust_rag.generation import LLMResponse
from antitrust_rag.service import RAGService
from antitrust_rag.sources import RawDocument, load_sample_dir
from antitrust_rag.store.memory import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"


class StubLLM:
    """Stands in for Claude: answers by citing the first source it was given. No network."""

    model = "claude-sonnet-5-5"

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        self.calls.append((system, user))
        if self.reply is not None:
            text = self.reply
        else:
            first = re.search(r'citation="([^"]+)"', user)
            assert first, "prompt must contain at least one cited source"
            text = f"According to the sources, this is regulated [{first.group(1)}]."
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=len(user) // 4,
            output_tokens=len(text) // 4,
            latency_ms=1.0,
        )


@pytest.fixture(scope="session")
def sample_docs() -> list[RawDocument]:
    return load_sample_dir(SAMPLE_DIR)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # never pick up a developer's local .env during tests
        store="memory",
        embedder="fake",
        sample_dir=str(SAMPLE_DIR),
        candidate_k=20,
    )


@pytest.fixture
def service(settings: Settings, sample_docs: list[RawDocument]) -> RAGService:
    svc = RAGService(
        InMemoryStore(settings.embedding_dim),
        FakeEmbedder(settings.embedding_dim),
        llm=StubLLM(),
        settings=settings,
    )
    svc.ingest_documents(sample_docs)
    return svc
