"""Bootstrap confidence intervals over per-query scores. Pure Python, seeded, offline.

With a few hundred queries, a one- or two-point gap between two modes can be an accident
of which queries were sampled. The paired interval below says how large that accident could be.

Limitation worth stating: resampling queries captures query-sampling noise only. It says
nothing about variation from other sources (a different corpus, model version, or hardware).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from metrics import percentile


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def bootstrap_mean(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 0, confidence: float = 0.95
) -> Interval:
    """Percentile-bootstrap interval for the mean. Deterministic for a given seed."""
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    n = len(values)
    rng = random.Random(seed)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(resamples)]
    tail = (1 - confidence) / 2 * 100
    return Interval(sum(values) / n, percentile(means, tail), percentile(means, 100 - tail))


def paired_difference(
    baseline: Mapping[str, float], candidate: Mapping[str, float], **kwargs
) -> Interval:
    """Interval for mean(candidate - baseline) over the same queries.

    Pairing matters: both modes are scored on the same queries, so a hard query is hard
    for both. Comparing two independent intervals would throw that information away.
    """
    if baseline.keys() != candidate.keys():
        raise ValueError("paired comparison needs identical query ids in both score sets")
    diffs = [candidate[q] - baseline[q] for q in sorted(baseline)]
    return bootstrap_mean(diffs, **kwargs)
