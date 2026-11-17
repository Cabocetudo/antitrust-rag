import numpy as np
import pytest

from antitrust_rag.chunking import chunk_document
from antitrust_rag.embeddings import FakeEmbedder
from antitrust_rag.models import Chunk, LegalUnit, ScoredChunk
from antitrust_rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from antitrust_rag.store.memory import InMemoryStore


def _chunk(art: str, text: str = "x", source: str = "s") -> Chunk:
    return Chunk(source_id=source, source_title="S", unit=LegalUnit(art), text=f"{art} {text}")


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    e = FakeEmbedder(64)
    a, b = e.embed_query("acordo de leniência"), e.embed_query("acordo de leniência")
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    near = float(a @ e.embed_query("acordo de leniência com o Cade"))
    far = float(a @ e.embed_query("abuse of a dominant position"))
    assert near > far


def test_rrf_math_and_rank_bookkeeping() -> None:
    a, b, c = _chunk("1"), _chunk("2"), _chunk("3")
    fused = reciprocal_rank_fusion(
        {
            "lexical": [ScoredChunk(a, 9), ScoredChunk(b, 5)],
            "vector": [ScoredChunk(b, 0.9), ScoredChunk(c, 0.8)],
        },
        k=60,
    )
    assert [f.chunk.unit.article for f in fused] == ["2", "1", "3"]  # b appears in both lists
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].ranks == {"lexical": 2, "vector": 1}


def test_rrf_top_n() -> None:
    lists = {"x": [ScoredChunk(_chunk(str(i)), 1.0) for i in range(10)]}
    assert len(reciprocal_rank_fusion(lists, top_n=3)) == 3


def test_memory_store_bm25_and_delete() -> None:
    store = InMemoryStore(dim=16)
    chunks = [
        _chunk("1", "leniência acordo cartel", "a"),
        _chunk("2", "fusão aquisição concentração", "a"),
        _chunk("3", "leniência", "b"),
    ]
    store.upsert(chunks, FakeEmbedder(16).embed_documents([c.text for c in chunks]))
    top = store.lexical_search("acordo de leniência", k=3)
    assert top[0].chunk.unit.article == "1"
    assert {r.chunk.unit.article for r in store.lexical_search("leniência", 5, ["b"])} == {"3"}
    assert store.delete_source("a") == 2
    assert store.count() == 1
    # upsert is idempotent by chunk id
    store.upsert(chunks[2:], FakeEmbedder(16).embed_documents([chunks[2].text]))
    assert store.count() == 1


def test_hybrid_retriever_end_to_end(sample_docs) -> None:
    emb = FakeEmbedder(128)
    store = InMemoryStore(128)
    for d in sample_docs:
        cs = chunk_document(d)
        store.upsert(cs, emb.embed_documents([c.text for c in cs]))
    r = HybridRetriever(store, emb, candidate_k=20)

    res = r.retrieve("acordo de leniência requisitos Superintendência-Geral", top_k=5)
    assert res.chunks[0].chunk.unit.article == "86"
    assert {"lexical_ms", "embed_ms", "vector_ms"} <= set(res.timings_ms)
    assert any(set(c.ranks) == {"lexical", "vector"} for c in res.chunks)

    only_eu = r.retrieve("abuse of a dominant position", 5, sources=["tfeu-102"])
    assert {c.chunk.source_id for c in only_eu.chunks} == {"tfeu-102"}

    lex = r.retrieve("posição dominante 20%", 3, mode="lexical")
    assert all(set(c.ranks) == {"lexical"} for c in lex.chunks)

    with pytest.raises(ValueError):
        r.retrieve("x y z", 3, mode="nope")


class ReverseReranker:
    def rerank(self, query, candidates, top_k):
        return list(reversed(candidates))[:top_k]


def test_reranker_is_applied_on_the_fused_pool(sample_docs) -> None:
    emb = FakeEmbedder(64)
    store = InMemoryStore(64)
    cs = chunk_document(sample_docs[0])
    store.upsert(cs, emb.embed_documents([c.text for c in cs]))
    plain = HybridRetriever(store, emb).retrieve("multa reincidência", 3, rerank=False)
    rr = HybridRetriever(store, emb, reranker=ReverseReranker(), rerank_pool=5)
    out = rr.retrieve("multa reincidência", 3)
    assert "rerank_ms" in out.timings_ms
    assert [c.chunk.chunk_id for c in out.chunks] != [c.chunk.chunk_id for c in plain.chunks]
