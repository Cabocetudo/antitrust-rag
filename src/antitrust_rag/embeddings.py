"""Pluggable text embedders.

* :class:`SentenceTransformerEmbedder` — default; a local multilingual model
  (``intfloat/multilingual-e5-small``, 384 dims) so Portuguese law and English questions share one
  vector space, with no per-call API cost.
* :class:`FakeEmbedder` — deterministic hashed bag-of-words vectors. No model download, no network,
  fast enough for unit tests and CI. It captures lexical overlap only, so it is *not* a semantic
  model and must never be used to judge retrieval quality.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Protocol, runtime_checkable

import numpy as np

from antitrust_rag.textutil import tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FakeEmbedder:
    """Feature-hashing embedder: each token (and token bigram) bumps a hashed dimension."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"fake-hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
        for tok in feats:
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model. E5 models expect ``query:`` / ``passage:`` prefixes."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        # sentence-transformers renamed this method in newer releases; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim() or 0)
        self.batch_size = batch_size
        self._e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        q = f"query: {text}" if self._e5 else text
        return np.asarray(
            self._model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )


def build_embedder(kind: str, model_name: str, dim: int) -> Embedder:
    if kind == "fake":
        return FakeEmbedder(dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown embedder {kind!r}")
