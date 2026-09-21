"""Retrieval-quality and latency metrics. Pure functions over ranked id lists.

Definitions follow trec_eval's conventions so numbers are comparable with published
BEIR results: linear gain, 1/log2(rank + 1) discount, cutoff applied before scoring.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

Judgments = Mapping[str, int]  # document id -> relevance grade (> 0 means relevant)


def _relevant(judgments: Judgments) -> set[str]:
    return {doc_id for doc_id, grade in judgments.items() if grade > 0}


def dcg(grades: Sequence[int], k: int) -> float:
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], judgments: Judgments, k: int = 10) -> float:
    actual = dcg([max(judgments.get(doc_id, 0), 0) for doc_id in ranked], k)
    ideal = dcg(sorted((g for g in judgments.values() if g > 0), reverse=True), k)
    return actual / ideal if ideal else 0.0


def recall_at_k(ranked: Sequence[str], judgments: Judgments, k: int = 10) -> float:
    relevant = _relevant(judgments)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], judgments: Judgments, k: int = 10) -> float:
    relevant = _relevant(judgments)
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def per_query_scores(
    runs: Mapping[str, Sequence[str]], qrels: Mapping[str, Judgments], k: int = 10
) -> dict[str, dict[str, float]]:
    """Metric -> {query id: score}. Kept per query so paired comparisons stay possible.

    Strict on purpose: a query with judgments but no run is an error, not a zero and
    not a skip. Silently dropping it would inflate every average.
    """
    missing = sorted(set(qrels) - set(runs))
    if missing:
        raise ValueError(f"{len(missing)} judged queries have no run, e.g. {missing[:3]}")
    return {
        f"ndcg@{k}": {q: ndcg_at_k(runs[q], qrels[q], k) for q in qrels},
        f"recall@{k}": {q: recall_at_k(runs[q], qrels[q], k) for q in qrels},
        f"mrr@{k}": {q: reciprocal_rank(runs[q], qrels[q], k) for q in qrels},
    }


def evaluate(
    runs: Mapping[str, Sequence[str]], qrels: Mapping[str, Judgments], k: int = 10
) -> dict[str, float]:
    """Mean of each metric over every judged query."""
    scores = per_query_scores(runs, qrels, k)
    return {name: sum(by_query.values()) / len(by_query) for name, by_query in scores.items()}


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (the numpy default), p in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(millis: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile(millis, 50),
        "p95_ms": percentile(millis, 95),
        "mean_ms": sum(millis) / len(millis),
    }
