"""Integration test against a real PostgreSQL + pgvector.

Skipped unless ``ANTITRUST_RAG_TEST_DSN`` is set, e.g.:

    docker compose up -d db
    ANTITRUST_RAG_TEST_DSN=postgresql://rag:rag@localhost:5432/antitrust_rag pytest -m postgres
"""

import os

import pytest

from antitrust_rag.chunking import chunk_document
from antitrust_rag.embeddings import FakeEmbedder
from antitrust_rag.retrieval import HybridRetriever

DSN = os.environ.get("ANTITRUST_RAG_TEST_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="ANTITRUST_RAG_TEST_DSN not set"),
]


@pytest.fixture
def pg_store():
    import psycopg

    from antitrust_rag.store.postgres import PostgresStore

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
    store = PostgresStore(DSN, dim=64)
    store.init_schema()
    return store


def test_postgres_roundtrip_and_hybrid_search(pg_store, sample_docs) -> None:
    emb = FakeEmbedder(64)
    total = 0
    for d in sample_docs:
        cs = chunk_document(d)
        total += pg_store.upsert(cs, emb.embed_documents([c.text for c in cs]))
    assert pg_store.count() == total
    assert pg_store.ping()

    lex = pg_store.lexical_search("acordo de leniência requisitos", 5)
    assert lex and lex[0].chunk.unit.article == "86"
    # accent-insensitive: "leniencia" matches "leniência"
    assert pg_store.lexical_search("leniencia", 3)

    vec = pg_store.vector_search(emb.embed_query("abuse of a dominant position"), 3, ["tfeu-102"])
    assert [v.chunk.source_id for v in vec] == ["tfeu-102"]

    res = HybridRetriever(pg_store, emb).retrieve("presume-se posição dominante 20%", 3)
    assert "36" in [c.chunk.unit.article for c in res.chunks]
    assert any(set(c.ranks) == {"lexical", "vector"} for c in res.chunks)

    # re-ingest of a source replaces its rows instead of duplicating them
    lei = next(d for d in sample_docs if d.spec.source_id == "lei-12529-2011")
    removed = pg_store.delete_source("lei-12529-2011")
    cs = chunk_document(lei)
    pg_store.upsert(cs, emb.embed_documents([c.text for c in cs]))
    assert removed == len(cs)
    assert pg_store.count() == total
