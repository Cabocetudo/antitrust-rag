"""Command-line entrypoints: corpus ingestion and offline retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from antitrust_rag.config import Settings
from antitrust_rag.evaluation import load_eval_set, reciprocal_rank, summarize
from antitrust_rag.logging_setup import configure_logging
from antitrust_rag.service import build_service
from antitrust_rag.sources import SOURCES, fetch_source, write_sample


def ingest_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download (or load) legal texts and index them.")
    p.add_argument(
        "--origin",
        choices=["remote", "sample"],
        default="remote",
        help="remote: download official texts; sample: use data/sample (offline)",
    )
    p.add_argument("--sources", nargs="*", choices=sorted(SOURCES), default=None)
    p.add_argument(
        "--save-raw",
        type=Path,
        default=None,
        help="also write the downloaded text lines to this directory (data/raw by convention)",
    )
    p.add_argument("--download-only", action="store_true", help="fetch + save, do not index")
    args = p.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level)

    if args.origin == "remote" and (args.save_raw or args.download_only):
        out_dir = args.save_raw or Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid in args.sources or sorted(SOURCES):
            url, lines = fetch_source(SOURCES[sid])
            write_sample(out_dir / f"{sid}.txt", SOURCES[sid], url, lines)
            print(f"saved {sid}: {len(lines)} lines from {url}")
        if args.download_only:
            return 0

    service = build_service(settings)
    if args.origin == "sample":
        written = service.ingest_sample(Path(settings.sample_dir))
    else:
        written = service.ingest_remote(args.sources)
    print(json.dumps({"chunks_written": written, "total": service.store.count()}, indent=2))
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieval eval: recall@k and MRR per mode.")
    p.add_argument("--eval-set", type=Path, default=Path("eval/eval_set.jsonl"))
    p.add_argument("--modes", nargs="*", default=["lexical", "vector", "hybrid"])
    p.add_argument("--rerank", action="store_true", help="also evaluate hybrid + cross-encoder")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ingest-sample", action="store_true", help="index data/sample first")
    p.add_argument("--json", type=Path, default=None, help="write the report as JSON here")
    p.add_argument("--verbose", action="store_true", help="print per-question first-hit rank")
    args = p.parse_args(argv)

    settings = Settings()
    if args.rerank:
        settings.rerank = True
    configure_logging("WARNING")
    service = build_service(settings)
    if args.ingest_sample or service.store.count() == 0:
        service.ingest_sample(Path(settings.sample_dir))

    items = load_eval_set(args.eval_set)
    runs = [(m, False) for m in args.modes] + ([("hybrid", True)] if args.rerank else [])
    report: dict[str, dict[str, float]] = {}
    for mode, rr in runs:
        per_q = []
        for it in items:
            res = service.search(it.question, args.top_k, mode=mode, rerank=rr)
            per_q.append((res.chunks, it.expected))
            if args.verbose:
                rank = reciprocal_rank(res.chunks, it.expected)
                first = round(1 / rank) if rank else None
                print(f"  [{mode}{'+rerank' if rr else ''}] {it.qid} first relevant rank={first}")
        report[mode + ("+rerank" if rr else "")] = summarize(per_q)

    header = {
        "embedder": service.embedder.name,
        "store": settings.store,
        "questions": len(items),
        "chunks_indexed": service.store.count(),
    }
    print(json.dumps(header))
    cols = list(next(iter(report.values())).keys())
    print(f"{'mode':<16}" + "".join(f"{c:>11}" for c in cols))
    for mode, metrics in report.items():
        print(f"{mode:<16}" + "".join(f"{metrics[c]:>11.3f}" for c in cols))
    if args.json:
        args.json.write_text(json.dumps({**header, "results": report}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(eval_main())
