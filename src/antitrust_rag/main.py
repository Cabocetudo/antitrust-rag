"""ASGI entrypoint: ``uvicorn antitrust_rag.main:app``."""

from antitrust_rag.api import create_app

app = create_app()
