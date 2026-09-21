"""Ingest SciFact into Elasticsearch for one semantic model.

    python run_ingest.py --model e5        # dense multilingual-e5-small
    python run_ingest.py --model elser     # sparse ELSER

Safe to interrupt and re-run: documents already indexed are skipped.
"""
from __future__ import annotations

import argparse
import time

from elasticsearch import Elasticsearch

from config import Config, positive_int
from dataset import load_scifact
from indexing import MODELS, ensure_index, index_name, ingest
from resilience import wait_for_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--batch-size", type=positive_int, default=20)
    parser.add_argument(
        "--limit", type=positive_int, default=None,
        help="index only the first N documents (N >= 1). For smoke tests: a subset is NOT a valid benchmark corpus.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config.from_env()
    if not cfg.es_password:
        raise SystemExit("ES_PASSWORD is not set (see .env.example)")
    client = Elasticsearch(cfg.es_url, basic_auth=(cfg.es_user, cfg.es_password), request_timeout=300)

    model = MODELS[args.model]
    name = index_name(cfg, model)
    corpus = load_scifact(cfg.data_dir).corpus
    docs = corpus if args.limit is None else corpus[: args.limit]
    subset = "" if args.limit is None else " (SUBSET, not a valid benchmark corpus)"
    print(f"{len(docs)} documents -> {name} on {cfg.es_url}{subset}")

    created = ensure_index(client, name, model)
    print(f"index {'created' if created else 'exists, resuming'}")
    print(f"waiting for {model.inference_id} ...", flush=True)
    print(f"model ready after {wait_for_inference(client, model.inference_id):.0f}s", flush=True)

    started = time.monotonic()

    def progress(result, total):
        done = result.indexed + result.skipped + len(result.failed)
        elapsed = time.monotonic() - started
        rate = result.indexed / elapsed if result.indexed else 0.0
        eta = f"{(total - done) / rate / 60:.0f} min" if rate else "?"
        print(
            f"{done:>5}/{total} | indexed {result.indexed} skipped {result.skipped} "
            f"failed {len(result.failed)} | {rate:.2f} docs/s | eta {eta}",
            flush=True,
        )

    result = ingest(client, name, docs, batch_size=args.batch_size, on_progress=progress)
    client.indices.refresh(index=name)
    count = client.count(index=name)["count"]
    print(f"\nindexed {result.indexed}, skipped {result.skipped}, failed {len(result.failed)} "
          f"in {(time.monotonic() - started) / 60:.1f} min; index holds {count} documents")
    for doc_id, reason in list(result.failed.items())[:10]:
        print(f"  FAILED {doc_id}: {reason}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
