"""PostgreSQL + pgvector store.

* Dense search: ``embedding <=> query`` (cosine distance) served by an HNSW index.
* Lexical search: a stored ``tsvector`` (``simple`` config over accent-stripped, lowercased text)
  with a GIN index, ranked by ``ts_rank_cd``. Terms are OR-ed so a natural-language question does
  not need every word to match. ``ts_rank_cd`` is a cover-density ranking, *not* true BM25 (no
  corpus-wide IDF); the in-memory store implements real BM25. See README "trade-offs".
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from antitrust_rag.models import Chunk, LegalUnit, ScoredChunk
from antitrust_rag.textutil import content_terms, normalize_text

logger = logging.getLogger(__name__)


def schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    source_title    TEXT NOT NULL,
    article         TEXT NOT NULL,
    paragraph       TEXT,
    item            TEXT,
    subitem         TEXT,
    citation        TEXT NOT NULL,
    citation_style  TEXT NOT NULL,
    heading_path    TEXT[] NOT NULL DEFAULT '{{}}',
    url             TEXT,
    language        TEXT NOT NULL,
    content         TEXT NOT NULL,
    lexical_text    TEXT NOT NULL,
    embedding       vector({dim}) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('simple', lexical_text)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source_id);
"""


_COLUMNS = (
    "id, source_id, source_title, article, paragraph, item, subitem, citation_style, "
    "heading_path, url, language, content"
)


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        source_id=row["source_id"],
        source_title=row["source_title"],
        unit=LegalUnit(row["article"], row["paragraph"], row["item"], row["subitem"]),
        text=row["content"],
        heading_path=list(row["heading_path"] or []),
        url=row["url"],
        language=row["language"],
        citation_style=row["citation_style"],
    )


class PostgresStore:
    def __init__(self, dsn: str, dim: int, ef_search: int = 64) -> None:
        self.dsn = dsn
        self.dim = dim
        self.ef_search = ef_search

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        # One short-lived connection per call keeps the demo simple; a real deployment would
        # use psycopg_pool.ConnectionPool (see README roadmap).
        with psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5) as conn:
            register_vector(conn)
            yield conn

    def init_schema(self) -> None:
        with psycopg.connect(self.dsn, connect_timeout=5, autocommit=True) as conn:
            conn.execute(schema_sql(self.dim))
        logger.info("schema ready", extra={"dim": self.dim})

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        rows = [
            (
                c.chunk_id,
                c.source_id,
                c.source_title,
                c.unit.article,
                c.unit.paragraph,
                c.unit.item,
                c.unit.subitem,
                c.citation,
                c.citation_style,
                c.heading_path,
                c.url,
                c.language,
                c.text,
                normalize_text(c.text),
                np.asarray(e, dtype=np.float32),
            )
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, source_id, source_title, article, paragraph, item,
                    subitem, citation, citation_style, heading_path, url, language, content,
                    lexical_text, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,
                    lexical_text = EXCLUDED.lexical_text, embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return len(rows)

    def delete_source(self, source_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE source_id = %s", [source_id])
            return cur.rowcount

    def vector_search(
        self, query_vec: np.ndarray, k: int, sources: list[str] | None = None
    ) -> list[ScoredChunk]:
        q = np.asarray(query_vec, dtype=np.float32)
        where, params = ("WHERE source_id = ANY(%s)", [sources]) if sources else ("", [])
        with self._conn() as conn, conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
            rows = conn.execute(
                f"""
                SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS score
                FROM chunks {where}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                [q, *params, q, k],
            ).fetchall()
        return [ScoredChunk(_row_to_chunk(r), float(r["score"])) for r in rows]

    def lexical_search(
        self, query: str, k: int, sources: list[str] | None = None
    ) -> list[ScoredChunk]:
        terms = sorted(set(content_terms(query)))
        if not terms:
            return []
        tsquery = " | ".join(terms)  # tokens are \w+ only, so this is injection-safe for to_tsquery
        src_filter, params = ("AND source_id = ANY(%s)", [sources]) if sources else ("", [])
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT {_COLUMNS}, ts_rank_cd(tsv, q, 1) AS score
                FROM chunks, to_tsquery('simple', %s) AS q
                WHERE tsv @@ q {src_filter}
                ORDER BY score DESC
                LIMIT %s
                """,
                [tsquery, *params, k],
            ).fetchall()
        return [ScoredChunk(_row_to_chunk(r), float(r["score"])) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False
