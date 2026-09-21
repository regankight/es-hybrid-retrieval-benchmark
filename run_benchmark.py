"""Run the benchmark against a built index and write artifacts/<tag>/.

    python run_benchmark.py --tag dev1 --modes bm25 semantic rrf
    python run_benchmark.py --tag release-1                     # all four modes, both phases

Two independent phases (see latency.py for why they must not share queries):
  quality  rank the scored test queries with each mode and score them
  latency  time each mode on its own disjoint held-out queries: cold, then cached

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
from latency import LatencySettings, run_latency
from resilience import wait_for_inference
from retrieval import DEFAULT_RERANK_INFERENCE_ID, MODES

REPO = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="run name; becomes artifacts/<tag>/")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--phases", nargs="+", choices=("quality", "latency"), default=["quality", "latency"])
    parser.add_argument("--model", choices=sorted(MODELS), default="e5")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--rerank-field", default="semantic")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stability-sample", type=int, default=50)
    parser.add_argument("--latency-queries", type=int, default=100, help="held-out queries per mode")
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
        rerank_field=args.rerank_field, stability_sample=args.stability_sample, warmup=args.warmup,
    )
    latency_settings = LatencySettings(per_mode=args.latency_queries, warmup=args.warmup)
    started = datetime.now(timezone.utc)
    quality = latency = None

    if "quality" in args.phases:
        print(f"quality: {args.modes} on {index} ({len(dataset.queries)} scored queries) ...", flush=True)
        quality = run_benchmark(client, index, args.modes, dataset.queries, dataset.qrels, settings, out_dir)
    if "latency" in args.phases:
        print(f"latency: {args.latency_queries} held-out queries per mode, cold then cached "
              f"(pool of {len(dataset.heldout)}) ...", flush=True)
        latency = run_latency(client, index, args.modes, dataset.heldout, latency_settings, settings, out_dir)

    write_json(out_dir, "index.json", client.indices.get(index=index).body)
    write_json(out_dir, "manifest.json", {
        "tag": args.tag,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "phases": args.phases,
        "index": index,
        "index_documents": index_docs,
        "modes": args.modes,
        "settings": asdict(settings),
        "latency_settings": asdict(latency_settings),
        "dataset": {
            "name": "BEIR/SciFact", "split": "test", "source": SCIFACT_URL, "archive_md5": SCIFACT_MD5,
            "corpus_documents": len(dataset.corpus), "scored_queries": len(dataset.queries),
            "heldout_pool": len(dataset.heldout),
        },
        "models": {"semantic": model.inference_id, "rerank": DEFAULT_RERANK_INFERENCE_ID},
        "code": version,
        "environment": collect_environment(client),
    })

    if quality:
        print(f"\nQUALITY  ({len(dataset.queries)} queries)")
        print(f"  {'mode':<11} {'nDCG@'+str(args.k):>8} {'recall@'+str(args.k):>10} {'MRR@'+str(args.k):>8}")
        for mode, m in quality.items():
            print(f"  {mode:<11} {m[f'ndcg@{args.k}']:>8.4f} {m[f'recall@{args.k}']:>10.4f} {m[f'mrr@{args.k}']:>8.4f}")
        unstable = {m: len(s["unstable_query_ids"]) for m, s in quality.items() if s["unstable_query_ids"]}
        if unstable:
            print(f"  NOTE: rankings changed on re-run for {unstable}")
    if latency:
        print(f"\nLATENCY  ({args.latency_queries} held-out queries per mode; p50 with 95% CI, p95, in ms)")
        print(f"  {'mode':<11} {'cold p50':>9} {'95% CI':>15} {'cold p95':>9} {'cached p50':>11} {'speedup':>8}")
        for mode, m in latency.items():
            c, w = m["cold"], m["warm"]
            ci = f"[{c['p50_ci_low_ms']:.0f}, {c['p50_ci_high_ms']:.0f}]"
            print(f"  {mode:<11} {c['p50_ms']:>9.0f} {ci:>15} {c['p95_ms']:>9.0f} {w['p50_ms']:>11.0f} "
                  f"{m['cache_speedup_p50']:>7.1f}x")
    if latency:
        retried = {m: s["retried_queries"] for m, s in latency.items() if s["retried_queries"]}
        if retried:
            print(f"\nWARNING: latency queries needed retries {retried}. Those timings exclude the failed "
                  "attempts, so treat this run as INVALID for latency.")
    print(f"\nartifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
