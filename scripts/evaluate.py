"""Offline retrieval evaluation (recall@k, MRR) over eval/eval_set.jsonl.

RAG_STORE=memory RAG_EMBEDDER=fake python scripts/evaluate.py      # no downloads at all
RAG_STORE=memory python scripts/evaluate.py --rerank                # local e5 + cross-encoder
"""

import sys

from antitrust_rag.cli import eval_main

if __name__ == "__main__":
    sys.exit(eval_main())
