"""Run the benchmark against a built index and write artifacts/<tag>/.

    python run_benchmark.py --tag dev1 --modes bm25 semantic rrf
    python run_benchmark.py --tag release-1 --repeats 3      # all four modes, from a clean commit

Tags starting with "dev" are scratch runs (gitignored). Any other tag is meant to be
committed, so it is refused unless the working tree is clean: a published number must be
traceable to a commit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from elasticsearch import Elasticsearch

from artifacts import write_json
from benchmark import BenchmarkSettings, run_benchmark, verify_index_complete
from config import Config
from dataset import SCIFACT_MD5, SCIFACT_URL, load_scifact
from environment import code_version, collect_environment
from indexing import MODELS, index_name
from resilience import wait_for_inference
from retrieval import DEFAULT_RERANK_INFERENCE_ID, MODES

REPO = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="run name; becomes artifacts/<tag>/")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--model", choices=sorted(MODELS), default="e5")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--rerank-field", default="semantic")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--note", default="", help="free text recorded in the manifest, e.g. machine load")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    if not cfg.es_password:
        raise SystemExit("ES_PASSWORD is not set (see .env.example)")
    out_dir = REPO / "artifacts" / args.tag
    if (out_dir / "manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already holds a run; pick a new --tag or pass --overwrite")
    version = code_version(REPO)
    if not args.tag.startswith("dev") and version["dirty"]:
        raise SystemExit("working tree has uncommitted changes; commit first or use a dev* tag")

    client = Elasticsearch(cfg.es_url, basic_auth=(cfg.es_user, cfg.es_password), request_timeout=300)
    model = MODELS[args.model]
    index = index_name(cfg, model)
    dataset = load_scifact(cfg.data_dir)
    index_docs = verify_index_complete(client, index, len(dataset.corpus))

    # Wake each model that a mode will use, so cold starts never land inside a timed query.
    if any(mode != "bm25" for mode in args.modes):
        print(f"waking {model.inference_id} ...", flush=True)
        wait_for_inference(client, model.inference_id)
    if "rrf_rerank" in args.modes:
        print(f"waking {DEFAULT_RERANK_INFERENCE_ID} ...", flush=True)
        wait_for_inference(client, DEFAULT_RERANK_INFERENCE_ID, query="warm up")

    settings = BenchmarkSettings(
        k=args.k, window=args.window, rank_constant=args.rank_constant,
        rerank_field=args.rerank_field, repeats=args.repeats, warmup=args.warmup,
    )
    started = datetime.now(timezone.utc)
    print(f"running {args.modes} on {index} ({len(dataset.queries)} queries, repeats={args.repeats}) ...", flush=True)
    summary = run_benchmark(client, index, args.modes, dataset.queries, dataset.qrels, settings, out_dir)

    write_json(out_dir, "index.json", client.indices.get(index=index).body)
    write_json(out_dir, "manifest.json", {
        "tag": args.tag,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "index": index,
        "index_documents": index_docs,
        "modes": args.modes,
        "settings": asdict(settings),
        "dataset": {
            "name": "BEIR/SciFact", "split": "test", "source": SCIFACT_URL, "archive_md5": SCIFACT_MD5,
            "corpus_documents": len(dataset.corpus), "queries": len(dataset.queries),
        },
        "models": {"semantic": model.inference_id, "rerank": DEFAULT_RERANK_INFERENCE_ID},
        "code": version,
        "environment": collect_environment(client),
    })

    print(f"\n{'mode':<11} {'nDCG@'+str(args.k):>8} {'recall@'+str(args.k):>10} {'MRR@'+str(args.k):>8} {'p50 ms':>8} {'p95 ms':>8}")
    for mode, m in summary.items():
        print(f"{mode:<11} {m[f'ndcg@{args.k}']:>8.4f} {m[f'recall@{args.k}']:>10.4f} {m[f'mrr@{args.k}']:>8.4f} "
              f"{m['p50_ms']:>8.0f} {m['p95_ms']:>8.0f}")
    unstable = {m: len(s["unstable_query_ids"]) for m, s in summary.items() if s["unstable_query_ids"]}
    if unstable:
        print(f"\nNOTE: rankings changed between repeats for {unstable}")
    print(f"\nartifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
