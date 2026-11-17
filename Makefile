# Antitrust RAG — common tasks. Run `make help` for the list.
PY ?= python
DSN ?= postgresql://rag:rag@localhost:5432/antitrust_rag

.PHONY: help install install-ml lint format typecheck test test-pg up down logs ingest ingest-sample eval eval-offline run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package + dev tools (no ML deps; enough for tests)
	$(PY) -m pip install -e ".[dev]"

install-ml:  ## Also install sentence-transformers (local embeddings / reranker, pulls PyTorch)
	$(PY) -m pip install -e ".[dev,ml]"

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## Auto-format and fix lint issues
	ruff format .
	ruff check --fix .

typecheck:  ## mypy on the package
	mypy src

test:  ## Unit + API tests (offline: fake embedder, mocked Claude, no API key)
	$(PY) -m pytest

test-pg:  ## Integration test against the docker-compose Postgres + pgvector
	ANTITRUST_RAG_TEST_DSN=$(DSN) $(PY) -m pytest -m postgres

up:  ## Start Postgres + API with docker compose
	docker compose up -d --build

down:  ## Stop containers
	docker compose down

logs:  ## Tail API logs (structured JSON)
	docker compose logs -f api

ingest:  ## Download official texts and index them into Postgres (host run)
	RAG_DATABASE_URL=$(DSN) $(PY) scripts/ingest.py --origin remote

ingest-sample:  ## Index the bundled offline sample into Postgres (host run)
	RAG_DATABASE_URL=$(DSN) $(PY) scripts/ingest.py --origin sample

eval:  ## Retrieval eval with the local e5 model + reranker, in memory, on the sample corpus
	RAG_STORE=memory $(PY) scripts/evaluate.py --rerank

eval-offline:  ## Retrieval eval with the deterministic fake embedder (no downloads at all)
	RAG_STORE=memory RAG_EMBEDDER=fake $(PY) scripts/evaluate.py

run:  ## Run the API locally with auto-reload
	uvicorn antitrust_rag.main:app --reload --port 8000

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
