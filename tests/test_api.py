import pytest
from fastapi.testclient import TestClient

from antitrust_rag.api import create_app
from antitrust_rag.embeddings import FakeEmbedder
from antitrust_rag.generation import NOT_FOUND
from antitrust_rag.service import RAGService
from antitrust_rag.store.memory import InMemoryStore

from .conftest import StubLLM


@pytest.fixture
def client(service, settings):
    with TestClient(create_app(service=service, settings=settings)) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 40
    assert body["llm_configured"] is True
    assert body["embedder"].startswith("fake")
    assert "x-request-id" in r.headers


def test_ingest_sample_is_idempotent(client) -> None:
    before = client.get("/health").json()["chunks_indexed"]
    r = client.post("/ingest", json={"origin": "sample"})
    assert r.status_code == 200
    assert set(r.json()["chunks_written"]) == {"lei-12529-2011", "tfeu-101", "tfeu-102"}
    assert r.json()["total_chunks"] == before


def test_ingest_subset(settings) -> None:
    svc = RAGService(InMemoryStore(384), FakeEmbedder(384), settings=settings)
    with TestClient(create_app(service=svc, settings=settings)) as c:
        r = c.post("/ingest", json={"origin": "sample", "sources": ["tfeu-102"]})
        assert r.json()["chunks_written"] == {"tfeu-102": 1}
        assert r.json()["total_chunks"] == 1


def test_search(client) -> None:
    r = client.post(
        "/search",
        json={"query": "acordo de leniência Superintendência-Geral requisitos", "top_k": 3},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert results[0]["citation"].startswith("Lei 12.529/2011, Art. 86")
    assert {"lexical", "vector"} & set(results[0]["ranks"])


def test_search_validation_error(client) -> None:
    assert client.post("/search", json={"query": "x"}).status_code == 422
    assert client.post("/search", json={"query": "valid", "mode": "magic"}).status_code == 422
    r = client.post("/search", json={"query": "valid query", "sources": ["unknown"]})
    assert r.status_code == 422


def test_ask_returns_cited_answer_usage_and_uses_cache(client, service) -> None:
    payload = {"question": "Quais são os requisitos do acordo de leniência?", "top_k": 4}
    r1 = client.post("/ask", json=payload)
    assert r1.status_code == 200
    a = r1.json()
    assert a["grounded"] is True and a["cached"] is False
    assert a["citations"] and all(c["supported"] for c in a["citations"])
    assert a["usage"]["input_tokens"] > 0
    assert a["usage"]["estimated_cost_usd"] > 0
    assert "llm_ms" in a["timings_ms"]

    r2 = client.post("/ask", json=payload)
    assert r2.json()["cached"] is True
    assert r2.json()["answer"] == a["answer"]
    assert len(service.llm.calls) == 1  # second call never reached the LLM

    health = client.get("/health").json()["metrics"]
    assert health["llm_calls"] == 1 and health["cache_hits"] == 1


def test_ask_not_found(service, settings) -> None:
    service.llm = StubLLM(reply=NOT_FOUND)
    with TestClient(create_app(service=service, settings=settings)) as c:
        r = c.post("/ask", json={"question": "What is the capital of France?"})
    body = r.json()
    assert body["grounded"] is False
    assert body["answer"] == NOT_FOUND
    assert body["citations"] == []


def test_ask_without_llm_degrades_to_503(service, settings) -> None:
    service.llm = None
    with TestClient(create_app(service=service, settings=settings)) as c:
        r = c.post("/ask", json={"question": "Quais são as penas?"})
        assert r.status_code == 503
        assert "/search" in r.json()["detail"]
        assert c.post("/search", json={"query": "Quais são as penas?"}).status_code == 200


def test_ask_with_empty_index_skips_llm(settings) -> None:
    llm = StubLLM()
    svc = RAGService(InMemoryStore(384), FakeEmbedder(384), llm=llm, settings=settings)
    with TestClient(create_app(service=svc, settings=settings)) as c:
        r = c.post("/ask", json={"question": "Quais são as penas?"})
    assert r.json()["answer"] == NOT_FOUND
    assert llm.calls == []


def test_openapi_schema(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ingest", "/search", "/ask"} <= set(paths)
