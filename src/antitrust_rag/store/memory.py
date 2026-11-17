"""In-memory store: cosine similarity with numpy and Okapi BM25 over normalized tokens."""

from __future__ import annotations

import math
import threading
from collections import Counter

import numpy as np

from antitrust_rag.models import Chunk, ScoredChunk
from antitrust_rag.textutil import content_terms, tokenize


class InMemoryStore:
    def __init__(self, dim: int, k1: float = 1.5, b: float = 0.75) -> None:
        self.dim = dim
        self.k1, self.b = k1, b
        self._lock = threading.Lock()
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._df: Counter[str] = Counter()

    # -- writes ------------------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        with self._lock:
            for chunk, vec in zip(chunks, embeddings, strict=True):
                cid = chunk.chunk_id
                if cid in self._tf:
                    self._df.subtract(self._tf[cid].keys())
                toks = tokenize(chunk.text)
                tf = Counter(toks)
                self._chunks[cid] = chunk
                self._vecs[cid] = np.asarray(vec, dtype=np.float32)
                self._tf[cid] = tf
                self._len[cid] = len(toks)
                self._df.update(tf.keys())
        return len(chunks)

    def delete_source(self, source_id: str) -> int:
        with self._lock:
            ids = [cid for cid, c in self._chunks.items() if c.source_id == source_id]
            for cid in ids:
                self._df.subtract(self._tf[cid].keys())
                for d in (self._chunks, self._vecs, self._tf, self._len):
                    d.pop(cid, None)
            self._df = +self._df  # drop zero counts
        return len(ids)

    # -- reads -------------------------------------------------------------------------------
    def _ids(self, sources: list[str] | None) -> list[str]:
        if not sources:
            return list(self._chunks)
        allowed = set(sources)
        return [cid for cid, c in self._chunks.items() if c.source_id in allowed]

    def vector_search(
        self, query_vec: np.ndarray, k: int, sources: list[str] | None = None
    ) -> list[ScoredChunk]:
        ids = self._ids(sources)
        if not ids:
            return []
        mat = np.stack([self._vecs[i] for i in ids])
        q = np.asarray(query_vec, dtype=np.float32)
        denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
        sims = mat @ q / np.where(denom == 0, 1.0, denom)
        order = np.argsort(-sims)[:k]
        return [ScoredChunk(self._chunks[ids[i]], float(sims[i])) for i in order]

    def lexical_search(
        self, query: str, k: int, sources: list[str] | None = None
    ) -> list[ScoredChunk]:
        terms = content_terms(query)
        ids = self._ids(sources)
        if not terms or not ids:
            return []
        n = len(self._chunks)
        avgdl = sum(self._len.values()) / max(n, 1)
        scores: list[tuple[float, str]] = []
        for cid in ids:
            tf, dl, s = self._tf[cid], self._len[cid], 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scores.append((s, cid))
        scores.sort(reverse=True)
        return [ScoredChunk(self._chunks[cid], s) for s, cid in scores[:k]]

    def count(self) -> int:
        return len(self._chunks)

    def ping(self) -> bool:
        return True
