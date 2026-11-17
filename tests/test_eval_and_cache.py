from pathlib import Path

import pytest

from antitrust_rag.cache import TTLCache
from antitrust_rag.evaluation import load_eval_set, recall_at_k, reciprocal_rank, summarize
from antitrust_rag.models import Chunk, LegalUnit, ScoredChunk

from .conftest import ROOT


def _r(source: str, art: str) -> ScoredChunk:
    return ScoredChunk(Chunk(source, "T", LegalUnit(art), f"{source}{art}"), 1.0)


def test_recall_and_mrr() -> None:
    results = [_r("a", "1"), _r("a", "2"), _r("b", "3")]
    exp = frozenset({("a", "2"), ("b", "3")})
    assert recall_at_k(results, exp, 1) == 0.0
    assert recall_at_k(results, exp, 2) == 0.5
    assert recall_at_k(results, exp, 3) == 1.0
    assert reciprocal_rank(results, exp) == 0.5
    assert reciprocal_rank(results, frozenset({("z", "9")})) == 0.0
    s = summarize([(results, exp)], ks=(1, 3))
    assert s == {"recall@1": 0.0, "recall@3": 1.0, "mrr": 0.5}


def test_eval_set_is_well_formed() -> None:
    items = load_eval_set(ROOT / "eval" / "eval_set.jsonl")
    assert 10 <= len(items) <= 15
    assert len({i.qid for i in items}) == len(items)
    assert all(i.expected for i in items)
    valid_sources = {"lei-12529-2011", "tfeu-101", "tfeu-102"}
    assert all(s in valid_sources for i in items for s, _ in i.expected)


def test_eval_runs_offline_with_fake_embedder(service) -> None:
    items = load_eval_set(Path(ROOT / "eval" / "eval_set.jsonl"))
    per_q = [(service.search(i.question, 10).chunks, i.expected) for i in items]
    report = summarize(per_q)
    assert set(report) == {"recall@1", "recall@3", "recall@5", "mrr"}
    assert 0.0 < report["mrr"] <= 1.0


def test_ttl_cache_lru_and_expiry(monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("antitrust_rag.cache.time.monotonic", lambda: now[0])
    c = TTLCache(max_items=2, ttl_s=10)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # "a" becomes most recently used
    c.set("c", 3)  # evicts "b"
    assert c.get("b") is None and c.get("c") == 3
    now[0] += 11
    assert c.get("a") is None  # expired
    assert c.hits == 2 and c.misses == 2


def test_cache_key_is_order_insensitive_for_kwargs() -> None:
    assert TTLCache.make_key(a=1, b=[1, 2]) == TTLCache.make_key(b=[1, 2], a=1)
    with pytest.raises(TypeError):
        TTLCache.make_key(1)  # type: ignore[misc]
