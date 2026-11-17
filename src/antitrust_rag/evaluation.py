"""Retrieval evaluation: recall@k and MRR against hand-labelled expected articles.

A retrieved chunk is *relevant* when its (source_id, article) is in the question's expected set.
Chunk-level labels would be stricter, but article-level labels survive chunking changes, which is
what we want when comparing chunking or fusion strategies.

* recall@k — share of a question's expected articles that appear in the top-k chunks, averaged.
* MRR      — mean of 1 / rank of the first relevant chunk (0 if none in the retrieved list).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from antitrust_rag.models import ScoredChunk


@dataclass(frozen=True)
class EvalItem:
    qid: str
    question: str
    expected: frozenset[tuple[str, str]]
    lang: str = "en"


def load_eval_set(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            EvalItem(
                qid=row["id"],
                question=row["question"],
                expected=frozenset((e["source_id"], e["article"]) for e in row["expected"]),
                lang=row.get("lang", "en"),
            )
        )
    return items


def _key(sc: ScoredChunk) -> tuple[str, str]:
    return (sc.chunk.source_id, sc.chunk.unit.article)


def recall_at_k(results: list[ScoredChunk], expected: frozenset[tuple[str, str]], k: int) -> float:
    if not expected:
        return 0.0
    found = {_key(r) for r in results[:k]} & expected
    return len(found) / len(expected)


def reciprocal_rank(results: list[ScoredChunk], expected: frozenset[tuple[str, str]]) -> float:
    for rank, r in enumerate(results, start=1):
        if _key(r) in expected:
            return 1.0 / rank
    return 0.0


def summarize(
    per_question: list[tuple[list[ScoredChunk], frozenset[tuple[str, str]]]],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    n = max(len(per_question), 1)
    out = {f"recall@{k}": sum(recall_at_k(r, e, k) for r, e in per_question) / n for k in ks}
    out["mrr"] = sum(reciprocal_rank(r, e) for r, e in per_question) / n
    return {k: round(v, 3) for k, v in out.items()}
