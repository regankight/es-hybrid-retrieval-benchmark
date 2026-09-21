"""Quality benchmark: rank the scored queries with each mode, score them, write rankings.

Latency is deliberately NOT measured here. See latency.py for why: timings taken while
scoring quality are distorted by Elasticsearch's query-embedding cache.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from artifacts import write_json, write_rankings
from dataset import Query
from metrics import evaluate
from retrieval import MODES, run_query


class IndexIncomplete(RuntimeError):
    """The index does not hold the whole corpus, so results on it would not be comparable."""


@dataclass(frozen=True)
class BenchmarkSettings:
    k: int = 10
    window: int = 20
    # Ranked ids saved per query. Later stages (an offline reranker) need candidates deeper
    # than the metric cutoff; metrics are still cut at k, so nDCG@k does not change with depth.
    depth: int = 20
    rank_constant: int = 60
    rerank_field: str = "semantic"
    stability_sample: int = 50  # queries re-run once to confirm rankings are deterministic
    warmup: int = 5  # queries run first and discarded (connections, JIT)
    # Retries suit a cold-start blip. Against a saturated model each retry queues more work
    # behind the timed-out request, so the budget is small and the backoff long.
    attempts: int = 3
    retry_delay: float = 5.0

    def __post_init__(self) -> None:
        if not self.k <= self.depth <= self.window:
            raise ValueError(f"need k ({self.k}) <= depth ({self.depth}) <= window ({self.window})")


def request_options(settings: BenchmarkSettings, *, size: int | None = None) -> dict:
    return dict(
        size=settings.k if size is None else size,
        window=settings.window,
        rank_constant=settings.rank_constant,
        rerank_field=settings.rerank_field,
    )


def verify_index_complete(client, index: str, expected_docs: int) -> int:
    """Refuse to benchmark a partial index. A half-ingested index would score every
    mode on a smaller corpus, and the numbers would look plausible while being wrong."""
    count = client.count(index=index)["count"]
    if count != expected_docs:
        raise IndexIncomplete(
            f"{index} holds {count} documents, expected {expected_docs}; refusing to benchmark it"
        )
    return count


def measure_rankings(
    client,
    index: str,
    mode: str,
    queries: Sequence[Query],
    settings: BenchmarkSettings,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, list[str]], list[str]]:
    """Rank every query once. Returns the rankings plus the ids, among a re-run sample,
    whose ranking changed (approximate search can be non-deterministic; say so if it is)."""
    options = request_options(settings, size=settings.depth)

    def rank(query: Query) -> list[str]:
        return run_query(client, index, mode, query, attempts=settings.attempts,
                         base_delay=settings.retry_delay, sleep=sleep, clock=clock, **options).ranked_ids

    for query in queries[: settings.warmup]:
        rank(query)  # discarded on purpose

    rankings = {query.id: rank(query) for query in queries}
    sample = queries[: settings.stability_sample]
    unstable = sorted(query.id for query in sample if rank(query) != rankings[query.id])
    return rankings, unstable


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
    """Rank and score each mode, writing rankings/<mode>.json and metrics.json."""
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"unknown modes {unknown}; expected some of {MODES}")

    summary: dict[str, dict] = {}
    for mode in modes:
        rankings, unstable = measure_rankings(
            client, index, mode, queries, settings, sleep=sleep, clock=clock
        )
        write_rankings(out_dir, mode, rankings)
        summary[mode] = {**evaluate(rankings, qrels, settings.k), "unstable_query_ids": unstable}
    write_json(out_dir, "metrics.json", {"k": settings.k, "queries": len(queries), "modes": summary})
    return summary
