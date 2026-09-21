"""Run the benchmark against a built index and write artifacts/<tag>/.

    python run_benchmark.py --tag dev1 --modes bm25 semantic
    python run_benchmark.py --tag release-1          # BM25, dense, RRF; both phases
    python run_benchmark.py --tag x --modes bm25 semantic rrf rrf_rerank   # adds Elasticsearch's reranker

Two independent phases (see latency.py for why they must not share queries):
  quality  rank the scored test queries with each mode and score them
  latency  time each mode on its own disjoint held-out queries: cold, then cached

Tags starting with "dev" are scratch runs (gitignored). Any other tag is meant to be
committed, so it is refused unless the working tree is clean: a published number must be
traceable to a commit. A latency run is also refused if Elasticsearch has not restarted since
an earlier run embedded some of the same held-out queries, because its embedding cache would
still hold them (override with --allow-warm-cache, which is recorded in the manifest).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from elasticsearch import Elasticsearch

from artifacts import write_json
from benchmark import BenchmarkSettings, run_benchmark, verify_index_complete
from config import Config, positive_int
from dataset import SCIFACT_MD5, SCIFACT_URL, load_scifact
from environment import collect_environment, node_started_utc
from indexing import MODELS, index_name
from latency import LatencySettings, build_latency_plan, cache_conflicts, load_prior_runs, run_latency
from provenance import code_version, require_clean_for_release, reset_run_dir, resolve_run_dir
from resilience import wait_for_inference
from retrieval import DEFAULT_RERANK_INFERENCE_ID, EMBEDDING_QUERY_MODES, MODES

REPO = Path(__file__).resolve().parent
# Elasticsearch's own reranker is opt-in: it needs ML memory headroom that a small cluster may
# lack, and against a saturated model it times out and retries pile more work behind it.
DEFAULT_MODES = ("bm25", "semantic", "rrf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="run name; becomes artifacts/<tag>/")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--phases", nargs="+", choices=("quality", "latency"), default=["quality", "latency"])
    parser.add_argument("--model", choices=sorted(MODELS), default="e5")
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--window", type=positive_int, default=20)
    parser.add_argument("--depth", type=positive_int, default=None, help="ranked ids saved per query (default: window)")
    parser.add_argument("--rank-constant", type=positive_int, default=60)
    parser.add_argument("--rerank-field", default="semantic")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stability-sample", type=int, default=50)
    parser.add_argument("--latency-queries", type=positive_int, default=100, help="held-out queries per mode")
    parser.add_argument("--note", default="", help="free text recorded in the manifest, e.g. machine load")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run with this tag")
    parser.add_argument("--allow-warm-cache", action="store_true",
                        help="run latency even though Elasticsearch has not restarted since an earlier run used the same queries")
    return parser


def _speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}x"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:  # a bad combination of flags is a usage error, reported before any request is made
        settings = BenchmarkSettings(
            k=args.k, window=args.window, depth=args.window if args.depth is None else args.depth,
            rank_constant=args.rank_constant, rerank_field=args.rerank_field,
            stability_sample=args.stability_sample, warmup=args.warmup,
        )
    except ValueError as err:
        build_parser().error(str(err))
    latency_settings = LatencySettings(per_mode=args.latency_queries, warmup=args.warmup)

    cfg = Config.from_env()
    if not cfg.es_password:
        raise SystemExit("ES_PASSWORD is not set (see .env.example)")
    version = code_version(REPO)
    require_clean_for_release(args.tag, version)
    out_dir = resolve_run_dir(REPO / "artifacts", args.tag, overwrite=args.overwrite)

    client = Elasticsearch(cfg.es_url, basic_auth=(cfg.es_user, cfg.es_password), request_timeout=300)
    model = MODELS[args.model]
    index = index_name(cfg, model)
    dataset = load_scifact(cfg.data_dir)
    index_docs = verify_index_complete(client, index, len(dataset.corpus))

    cache_guard: dict = {"conflicts": [], "bypassed": False}
    if "latency" in args.phases:
        # Planned up front: fails fast if the held-out pool is too small, and tells us which
        # queries this run will embed so we can check none is already sitting in the cache.
        plan = build_latency_plan(dataset.heldout, args.modes, latency_settings)
        planned = {q.id for mode in args.modes if mode in EMBEDDING_QUERY_MODES for q in plan.slices[mode]}
        started = node_started_utc(client)
        conflicts = cache_conflicts(load_prior_runs(REPO / "artifacts", exclude_tag=args.tag), planned, started)
        cache_guard = {"elasticsearch_started_utc": started.isoformat(timespec="seconds"),
                       "conflicts": conflicts, "bypassed": bool(conflicts) and args.allow_warm_cache}
        if conflicts and not args.allow_warm_cache:
            raise SystemExit(
                f"Elasticsearch started at {started:%Y-%m-%d %H:%M:%S} UTC, before {', '.join(conflicts)} "
                "embedded some of these held-out queries, so its embedding cache still holds them and the "
                "cold pass would be flattered. Restart Elasticsearch (or pass --allow-warm-cache)."
            )

    # Wake each model that a mode will use, so cold starts never land inside a timed query.
    if any(mode != "bm25" for mode in args.modes):
        print(f"waking {model.inference_id} ...", flush=True)
        wait_for_inference(client, model.inference_id)
    if "rrf_rerank" in args.modes:
        print(f"waking {DEFAULT_RERANK_INFERENCE_ID} ...", flush=True)
        wait_for_inference(client, DEFAULT_RERANK_INFERENCE_ID, query="warm up")

    reset_run_dir(out_dir)  # only now: every precondition passed, so a failed start keeps the old run
    started_at = datetime.now(timezone.utc)
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
        "started_utc": started_at.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "phases": args.phases,
        "index": index,
        "index_documents": index_docs,
        "modes": args.modes,
        "settings": asdict(settings),
        "latency_settings": asdict(latency_settings),
        "cache_guard": cache_guard,
        "dataset": {
            "name": "BEIR/SciFact", "split": "test", "source": SCIFACT_URL, "archive_md5": SCIFACT_MD5,
            "corpus_documents": len(dataset.corpus), "scored_queries": len(dataset.queries),
            "heldout_pool": len(dataset.heldout),
        },
        "models": {"semantic": model.inference_id, "rerank": DEFAULT_RERANK_INFERENCE_ID},
        "code": version,
        "environment": collect_environment(client),
    })

    k = args.k
    if quality:
        print(f"\nQUALITY  ({len(dataset.queries)} queries)")
        print(f"  {'mode':<11} {'nDCG@'+str(k):>8} {'recall@'+str(k):>10} {'MRR@'+str(k):>8}")
        for mode, m in quality.items():
            print(f"  {mode:<11} {m[f'ndcg@{k}']:>8.4f} {m[f'recall@{k}']:>10.4f} {m[f'mrr@{k}']:>8.4f}")
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
                  f"{_speedup(m['cache_speedup_p50']):>8}")
        contaminated = {m: s["cached_like_cold_queries"] for m, s in latency.items() if s["cache_contaminated"]}
        if contaminated:
            print(f"\nWARNING: cold-pass queries ran at cached speed {contaminated}. Elasticsearch's embedding "
                  "cache already held them, so this run is INVALID for latency. Restart Elasticsearch to clear "
                  "the cache (or use unused held-out queries) and run again.")
        retried = {m: s["retried_queries"] for m, s in latency.items() if s["retried_queries"]}
        if retried:
            print(f"\nWARNING: latency queries needed retries {retried}. Those timings exclude the failed "
                  "attempts, so treat this run as INVALID for latency.")
        if "rrf_rerank" in latency and not latency["rrf_rerank"]["cache_check"]:
            print("\nNOTE: the cache-contamination check does not apply to rrf_rerank (its rerank inference "
                  "is not cached, so cold and cached look alike); the restart guard above is what protects it.")
    print(f"\nartifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
