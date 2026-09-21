"""Latency measurement that Elasticsearch's query-embedding cache cannot flatter.

Elasticsearch caches query embeddings. Measured here on this setup: the same query text
cost about 90 ms the first time and about 2 ms the second. Any design in which two modes
see the same text therefore rewards whichever runs second, and a median taken over repeats
quietly reports the cached figure as if it were the real one. An earlier version of this
benchmark did exactly that: RRF appeared faster than the semantic search it contains.

So the design removes the confound by construction:

- Each mode gets its own disjoint slice of held-out queries: real queries that are never
  scored and never seen by any other mode. Every mode pays full inference cost on its cold pass.
- A second pass over the same slice reports the cached figure explicitly, as a separate
  number, instead of averaging it in. The cache is a real production effect worth knowing.
- Modes are interleaved query by query, with the order rotating each round, so slow drift
  (thermal throttling, model autoscaling) lands on every mode evenly instead of on whichever
  happened to run last or first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from artifacts import write_json, write_latency
from benchmark import BenchmarkSettings, request_options
from dataset import Query
from metrics import latency_summary
from retrieval import run_query
from stats import bootstrap_median


@dataclass(frozen=True)
class LatencySettings:
    per_mode: int = 100  # held-out queries per mode; p95 from 100 samples is coarse
    warmup: int = 5  # shared queries run first, per mode, and discarded


@dataclass(frozen=True)
class LatencyPlan:
    warmup: list[Query]
    slices: dict[str, list[Query]]


@dataclass
class ModeLatency:
    query_ids: list[str] = field(default_factory=list)
    cold_wall_ms: list[float] = field(default_factory=list)
    cold_took_ms: list[float] = field(default_factory=list)
    warm_wall_ms: list[float] = field(default_factory=list)
    warm_took_ms: list[float] = field(default_factory=list)


def build_latency_plan(
    pool: Sequence[Query], modes: Sequence[str], settings: LatencySettings
) -> LatencyPlan:
    """Split the held-out pool into a warm-up set plus one disjoint slice per mode."""
    needed = settings.warmup + settings.per_mode * len(modes)
    if len(pool) < needed:
        raise ValueError(
            f"need {needed} held-out queries ({settings.warmup} warm-up + "
            f"{settings.per_mode} x {len(modes)} modes) but only {len(pool)} are available"
        )
    if len({q.text for q in pool}) != len(pool):
        raise ValueError("held-out pool contains duplicate query texts, which the cache would exploit")
    rest = pool[settings.warmup :]
    return LatencyPlan(
        warmup=list(pool[: settings.warmup]),
        slices={
            mode: list(rest[i * settings.per_mode : (i + 1) * settings.per_mode])
            for i, mode in enumerate(modes)
        },
    )


def rotated(modes: Sequence[str], round_index: int) -> list[str]:
    """The mode order for one round, shifted each round so every mode takes every
    position equally often."""
    shift = round_index % len(modes)
    return list(modes[shift:]) + list(modes[:shift])


def measure_latency(
    client,
    index: str,
    modes: Sequence[str],
    pool: Sequence[Query],
    settings: LatencySettings,
    options: BenchmarkSettings,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, ModeLatency]:
    plan = build_latency_plan(pool, modes, settings)
    request = request_options(options)

    def one(mode: str, query: Query):
        return run_query(client, index, mode, query, sleep=sleep, clock=clock, **request)

    for mode in modes:
        for query in plan.warmup:
            one(mode, query)  # discarded: connections and JIT; these texts are in no slice

    results = {mode: ModeLatency(query_ids=[q.id for q in plan.slices[mode]]) for mode in modes}
    for phase in ("cold", "warm"):  # the second pass sees text each mode has already embedded
        for round_index in range(settings.per_mode):
            for mode in rotated(modes, round_index):
                run = one(mode, plan.slices[mode][round_index])
                getattr(results[mode], f"{phase}_wall_ms").append(run.wall_ms)
                getattr(results[mode], f"{phase}_took_ms").append(run.took_ms)
    return results


def _phase(wall_ms: Sequence[float], took_ms: Sequence[float]) -> dict[str, float]:
    ci = bootstrap_median(list(wall_ms))
    return {
        "n": len(wall_ms),
        **latency_summary(wall_ms),
        "p50_ci_low_ms": ci.low,
        "p50_ci_high_ms": ci.high,
        "server_p50_ms": latency_summary(took_ms)["p50_ms"],
    }


def summarize_latency(latency: ModeLatency) -> dict:
    cold = _phase(latency.cold_wall_ms, latency.cold_took_ms)
    warm = _phase(latency.warm_wall_ms, latency.warm_took_ms)
    return {"cold": cold, "warm": warm, "cache_speedup_p50": cold["p50_ms"] / warm["p50_ms"]}


def run_latency(
    client,
    index: str,
    modes: Sequence[str],
    pool: Sequence[Query],
    settings: LatencySettings,
    options: BenchmarkSettings,
    out_dir: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, dict]:
    measured = measure_latency(client, index, modes, pool, settings, options, sleep=sleep, clock=clock)
    summary = {mode: summarize_latency(latency) for mode, latency in measured.items()}
    for mode, latency in measured.items():
        write_latency(out_dir, mode, {
            "slice_query_ids": latency.query_ids,
            "cold": {"wall_ms": latency.cold_wall_ms, "took_ms": latency.cold_took_ms},
            "warm": {"wall_ms": latency.warm_wall_ms, "took_ms": latency.warm_took_ms},
        })
    write_json(out_dir, "latency.json", {
        "per_mode": settings.per_mode,
        "warmup": settings.warmup,
        "order": "interleaved query by query, rotating each round",
        "modes": summary,
    })
    return summary
