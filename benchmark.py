"""Benchmark orchestration: measure each mode, score it, write artifacts."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from artifacts import write_json, write_runs
from dataset import Query
from metrics import evaluate, latency_summary
from retrieval import MODES, SearchRun, run_query


class IndexIncomplete(RuntimeError):
    """The index does not hold the whole corpus, so results on it would not be comparable."""


@dataclass(frozen=True)
class BenchmarkSettings:
    k: int = 10
    window: int = 20
    rank_constant: int = 60
    rerank_field: str = "semantic"
    repeats: int = 1  # full passes over the query set; latency is the per-query median
    warmup: int = 5  # queries run first and discarded, to wake models and warm caches


def verify_index_complete(client, index: str, expected_docs: int) -> int:
    """Refuse to benchmark a partial index. A half-ingested index would score every
    mode on a smaller corpus, and the numbers would look plausible while being wrong."""
    count = client.count(index=index)["count"]
    if count != expected_docs:
        raise IndexIncomplete(
            f"{index} holds {count} documents, expected {expected_docs}; refusing to benchmark it"
        )
    return count


def measure_mode(
    client,
    index: str,
    mode: str,
    queries: Sequence[Query],
    settings: BenchmarkSettings,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[SearchRun], list[str]]:
    """Run every query. Returns the runs plus the ids of queries whose ranking changed
    between repeats (approximate search can be non-deterministic; that should be visible)."""
    options = dict(
        size=settings.k,
        window=settings.window,
        rank_constant=settings.rank_constant,
        rerank_field=settings.rerank_field,
    )

    def one(query: Query) -> SearchRun:
        return run_query(client, index, mode, query, sleep=sleep, clock=clock, **options)

    for query in queries[: settings.warmup]:
        one(query)  # discarded on purpose

    first: dict[str, list[str]] = {}
    took: dict[str, list[float]] = {}
    wall: dict[str, list[float]] = {}
    unstable: set[str] = set()
    for repeat in range(settings.repeats):
        for query in queries:
            run = one(query)
            if repeat == 0:
                first[query.id] = run.ranked_ids
            elif run.ranked_ids != first[query.id]:
                unstable.add(query.id)
            took.setdefault(query.id, []).append(run.took_ms)
            wall.setdefault(query.id, []).append(run.wall_ms)

    runs = [
        SearchRun(q.id, first[q.id], statistics.median(took[q.id]), statistics.median(wall[q.id]))
        for q in queries
    ]
    return runs, sorted(unstable)


def run_benchmark(
    client,
    index: str,
    modes: Sequence[str],
    queries: Sequence[Query],
    qrels: Mapping[str, Mapping[str, int]],
    settings: BenchmarkSettings,
    out_dir: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, dict]:
    """Measure and score each mode, writing runs/<mode>.json and metrics.json."""
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"unknown modes {unknown}; expected some of {MODES}")

    summary: dict[str, dict] = {}
    for mode in modes:
        runs, unstable = measure_mode(client, index, mode, queries, settings, sleep=sleep, clock=clock)
        write_runs(out_dir, mode, runs)
        quality = evaluate({r.query_id: r.ranked_ids for r in runs}, qrels, settings.k)
        client_ms = latency_summary([r.wall_ms for r in runs])
        server_ms = latency_summary([r.took_ms for r in runs])
        summary[mode] = {
            **quality,
            **client_ms,
            **{f"server_{name}": value for name, value in server_ms.items()},
            "unstable_query_ids": unstable,
        }
    write_json(out_dir, "metrics.json", {"k": settings.k, "queries": len(queries), "modes": summary})
    return summary
