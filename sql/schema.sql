-- Reference schema (generated from antitrust_rag.store.postgres.schema_sql(384)).
-- The app creates it automatically on startup; this file is for reading/review.
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
    heading_path    TEXT[] NOT NULL DEFAULT '{}',
    url             TEXT,
    language        TEXT NOT NULL,
    content         TEXT NOT NULL,
    lexical_text    TEXT NOT NULL,
    embedding       vector(384) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('simple', lexical_text)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source_id);
