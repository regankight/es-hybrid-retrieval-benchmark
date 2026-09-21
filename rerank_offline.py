"""Offline reranking of saved candidates with a local cross-encoder.

    .venv-rerank/bin/python rerank_offline.py --source artifacts/release-1 --tag rerank-1

Stage two of a two-stage retrieval pipeline: Elasticsearch supplies RRF's saved top-20
candidates per query (stage one, already captured in the source run's artifacts), and a
small cross-encoder re-scores each (query, passage) pair on this machine's CPU. Needs no
Elasticsearch, so it keeps working after the trial license behind the cluster expires.

This is NOT Elastic's reranker (`.rerank-v1`), which proved unusable on this cluster; it is a
different, much smaller model, so the result speaks about cross-encoder reranking in general.

The model code is imported lazily so everything else here can be tested without PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from artifacts import LOCAL_RERANK_MODE, read_rankings, write_json, write_rankings
from config import positive_int
from dataset import Document, Query, load_scifact
from metrics import evaluate, latency_summary
from provenance import code_version, require_clean_for_release, reset_run_dir, resolve_run_dir
from stats import bootstrap_median

MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"  # pinned commit, Apache-2.0
MAX_LENGTH = 512  # the model's position limit; the passage is truncated, never the query
MODE = LOCAL_RERANK_MODE

# score(query, passages) -> one relevance score per passage. Injectable so tests need no model.
Scorer = Callable[[str, Sequence[str]], list[float]]


class SourceError(ValueError):
    """The source run cannot supply what reranking needs."""


def rerank(candidates: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Order candidates by score, best first. Ties keep their incoming (RRF) order, so an
    uninformative reranker degrades to plain RRF instead of shuffling it."""
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [candidates[i] for i in order]


def rerank_queries(
    queries: Sequence[Query],
    candidates: Mapping[str, Sequence[str]],
    texts: Mapping[str, str],
    score: Scorer,
    *,
    warmup: int = 3,
    stability_sample: int = 50,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, list[str]], dict[str, float], list[str]]:
    """Rerank every query. Returns rankings, per-query compute time in ms, and the ids among
    a re-scored sample whose ranking changed (a stage that is not deterministic must say so)."""

    def order(query: Query) -> list[str]:
        ids = list(candidates[query.id])
        missing = [i for i in ids if i not in texts]
        if missing:
            raise SourceError(f"query {query.id}: candidates {missing[:3]} have no document text")
        return rerank(ids, score(query.text, [texts[i] for i in ids]))

    for query in queries[:warmup]:
        order(query)  # discarded: first calls pay one-off costs (allocation, kernel selection)

    rankings: dict[str, list[str]] = {}
    millis: dict[str, float] = {}
    for query in queries:
        started = clock()
        rankings[query.id] = order(query)
        millis[query.id] = (clock() - started) * 1000
    sample = queries[:stability_sample]
    unstable = sorted(q.id for q in sample if order(q) != rankings[q.id])
    return rankings, millis, unstable


def verify_source(source_rankings: Mapping[str, Sequence[str]], queries: Sequence[Query], candidates: int) -> None:
    missing = [q.id for q in queries if q.id not in source_rankings]
    if missing:
        raise SourceError(f"source run has no ranking for {len(missing)} scored queries, e.g. {missing[:3]}")
    shallow = [q for q, ranked in source_rankings.items() if len(ranked) < candidates]
    if shallow:
        raise SourceError(f"{len(shallow)} queries have fewer than {candidates} saved candidates; "
                          "re-run the source benchmark with a larger --depth")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_rerank(
    source: Path,
    out_dir: Path,
    queries: Sequence[Query],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Sequence[Document],
    score: Scorer,
    *,
    candidates: int = 20,
    k: int = 10,
    warmup: int = 3,
    stability_sample: int = 50,
    provenance: Mapping[str, object] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    source_rankings = read_rankings(source, "rrf")
    verify_source(source_rankings, queries, candidates)
    pool = {qid: ranked[:candidates] for qid, ranked in source_rankings.items()}
    texts = {doc.id: doc.passage() for doc in corpus}

    rankings, millis, unstable = rerank_queries(
        queries, pool, texts, score, warmup=warmup, stability_sample=stability_sample, clock=clock
    )
    write_rankings(out_dir, MODE, rankings)
    quality = evaluate(rankings, qrels, k)
    baseline = evaluate({q.id: pool[q.id] for q in queries}, qrels, k)  # RRF cut to the same pool

    per_query = [millis[q.id] for q in queries]
    interval = bootstrap_median(per_query)
    latency = {
        "stage": f"rerank {candidates} candidates per query with a local cross-encoder",
        "n": len(per_query),
        **latency_summary(per_query),
        "p50_ci_low_ms": interval.low,
        "p50_ci_high_ms": interval.high,
        "per_query_ms": millis,
    }
    write_json(out_dir, "latency.json", latency)
    write_json(out_dir, "metrics.json", {
        "k": k, "queries": len(queries), "candidates": candidates,
        "modes": {MODE: {**quality, "unstable_query_ids": unstable}},
        "rrf_same_pool": baseline,
    })
    write_json(out_dir, "manifest.json", {
        "source_run": str(source), "source_rankings_sha256": sha256_of(Path(source) / "rankings" / "rrf.json"),
        "candidates": candidates, "k": k, "warmup": warmup, "stability_sample": stability_sample,
        **(provenance or {}),
    })
    return {"quality": quality, "rrf": baseline, "latency": latency, "unstable": unstable}


def load_cross_encoder(threads: int) -> tuple[Scorer, dict]:
    """Load the pinned model on CPU with a fixed thread count. Imported lazily: needs the
    optional dependencies in requirements-rerank.txt."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, revision=MODEL_REVISION).eval()

    def score(query: str, passages: Sequence[str]) -> list[float]:
        with torch.inference_mode():
            batch = tokenizer([query] * len(passages), list(passages), padding=True,
                              truncation="only_second", max_length=MAX_LENGTH, return_tensors="pt")
            return model(**batch).logits.squeeze(-1).tolist()

    info = {
        "repo": MODEL_REPO, "revision": MODEL_REVISION, "license": "apache-2.0",
        "parameters": sum(p.numel() for p in model.parameters()), "max_length": MAX_LENGTH,
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    return score, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="benchmark run holding rankings/rrf.json")
    parser.add_argument("--tag", required=True, help="output run name: artifacts/<tag>/")
    parser.add_argument("--candidates", type=positive_int, default=20)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run with this tag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo = Path(__file__).resolve().parent
    version = code_version(repo)
    require_clean_for_release(args.tag, version)
    out_dir = resolve_run_dir(repo / "artifacts", args.tag, overwrite=args.overwrite)

    dataset = load_scifact(args.data_dir)
    score, model_info = load_cross_encoder(args.threads)
    reset_run_dir(out_dir)  # only after the model and data loaded: a failed start keeps the old run
    started = datetime.now(timezone.utc)
    print(f"reranking the top {args.candidates} RRF candidates for {len(dataset.queries)} queries "
          f"with {MODEL_REPO} on CPU ({model_info['torch_threads']} threads) ...", flush=True)

    result = run_rerank(
        args.source, out_dir, dataset.queries, dataset.qrels, dataset.corpus, score,
        candidates=args.candidates,
        provenance={
            "tag": args.tag, "note": args.note, "model": model_info, "code": version,
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "machine": platform.machine(), "cpu_count": os.cpu_count()},
        },
    )
    q, b, lat = result["quality"], result["rrf"], result["latency"]
    print(f"\n{'':<24} {'nDCG@10':>8} {'recall@10':>10} {'MRR@10':>8}")
    print(f"{'RRF (same pool)':<24} {b['ndcg@10']:>8.4f} {b['recall@10']:>10.4f} {b['mrr@10']:>8.4f}")
    print(f"{'RRF + local rerank':<24} {q['ndcg@10']:>8.4f} {q['recall@10']:>10.4f} {q['mrr@10']:>8.4f}")
    print(f"\nrerank stage per query (CPU): p50 {lat['p50_ms']:.0f} ms [{lat['p50_ci_low_ms']:.0f}, "
          f"{lat['p50_ci_high_ms']:.0f}], p95 {lat['p95_ms']:.0f} ms")
    if result["unstable"]:
        print(f"NOTE: rankings changed on re-score for {len(result['unstable'])} sampled queries")
    print(f"\nartifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
