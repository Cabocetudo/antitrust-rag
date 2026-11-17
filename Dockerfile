FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[ml]"

COPY data ./data
COPY eval ./eval
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 app && mkdir -p /app/.cache && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "antitrust_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
