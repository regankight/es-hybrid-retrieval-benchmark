import pytest

from artifacts import read_json, read_latency
from benchmark import BenchmarkSettings
from dataset import Query
from fakes import ScriptedSearchClient, StepClock
from latency import (
    LatencySettings,
    ModeLatency,
    build_latency_plan,
    measure_latency,
    rotated,
    run_latency,
    summarize_latency,
)

MODES = ["bm25", "semantic", "rrf"]
POOL = [Query(f"h{i}", f"held-out text {i}") for i in range(40)]
NO_SLEEP = lambda seconds: None  # noqa: E731


def mode_of(request):
    if "query" in request:
        return "bm25" if "multi_match" in request["query"] else "semantic"
    return "rrf"


class Recorder:
    """A search client that logs (mode, query text) for every call, in order."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.client = ScriptedSearchClient(self._hits)

    def _hits(self, text, request):
        self.calls.append((mode_of(request), text))
        return ["a"]


def measure(recorder, per_mode=3, warmup=2, step=0.01, pool=POOL):
    return measure_latency(
        recorder.client, "idx", MODES, pool, LatencySettings(per_mode=per_mode, warmup=warmup),
        BenchmarkSettings(), sleep=NO_SLEEP, clock=StepClock(step),
    )


def test_the_plan_reserves_warmup_and_gives_each_mode_a_disjoint_slice():
    plan = build_latency_plan(POOL, MODES, LatencySettings(per_mode=5, warmup=3))
    ids = [q.id for q in plan.warmup] + [q.id for s in plan.slices.values() for q in s]
    assert len(ids) == len(set(ids)) == 3 + 5 * 3
    assert all(len(s) == 5 for s in plan.slices.values())


def test_the_plan_refuses_a_pool_that_is_too_small():
    with pytest.raises(ValueError, match="held-out queries"):
        build_latency_plan(POOL[:10], MODES, LatencySettings(per_mode=5, warmup=3))


def test_the_plan_refuses_duplicate_texts_because_the_cache_would_exploit_them():
    pool = POOL[:30] + [Query("dup", POOL[0].text)]
    with pytest.raises(ValueError, match="duplicate"):
        build_latency_plan(pool, MODES, LatencySettings(per_mode=5, warmup=3))


def test_rotation_puts_every_mode_in_every_position_equally_often():
    positions = {m: [] for m in MODES}
    for round_index in range(len(MODES)):
        for position, mode in enumerate(rotated(MODES, round_index)):
            positions[mode].append(position)
    assert all(sorted(p) == [0, 1, 2] for p in positions.values())


def test_no_query_text_is_seen_by_two_modes_apart_from_the_shared_warmup():
    recorder = Recorder()
    measure(recorder, per_mode=3, warmup=2)
    warmup_texts = {q.text for q in POOL[:2]}
    seen: dict[str, list[str]] = {}
    for mode, text in recorder.calls:
        if text not in warmup_texts:
            seen.setdefault(text, []).append(mode)
    assert len(seen) == 3 * 3
    for modes in seen.values():
        assert len(set(modes)) == 1  # one mode only...
        assert len(modes) == 2  # ...seen once cold, once cached


def test_the_cold_pass_runs_before_the_cached_pass_and_modes_interleave_with_rotation():
    recorder = Recorder()
    measure(recorder, per_mode=3, warmup=0)
    order = [mode for mode, _ in recorder.calls]
    cold, warm = order[:9], order[9:]
    assert cold[:3] == ["bm25", "semantic", "rrf"]
    assert cold[3:6] == ["semantic", "rrf", "bm25"]  # rotated one place
    assert cold == warm  # the cached pass repeats the same interleaving
    texts = [t for _, t in recorder.calls]
    assert texts[:9] == texts[9:]  # and the same texts, second time around


def test_the_number_of_requests_is_exactly_what_the_plan_implies():
    recorder = Recorder()
    measure(recorder, per_mode=4, warmup=2)
    assert len(recorder.calls) == 2 * 3 + 2 * 4 * 3  # warm-up per mode + cold and cached passes


def test_timings_come_from_the_clock_for_every_query_and_both_passes():
    latency = measure(Recorder(), per_mode=3, warmup=1, step=0.02)
    for mode in MODES:
        assert latency[mode].cold_wall_ms == pytest.approx([20.0] * 3)
        assert latency[mode].warm_wall_ms == pytest.approx([20.0] * 3)
        assert len(latency[mode].query_ids) == 3


def test_summary_reports_cold_and_cached_separately_and_their_ratio():
    ml = ModeLatency(query_ids=list("abcde"), cold_wall_ms=[10, 20, 30, 40, 50],
                     cold_took_ms=[5] * 5, warm_wall_ms=[1] * 5, warm_took_ms=[1] * 5)
    got = summarize_latency(ml)
    assert got["cold"]["p50_ms"] == 30 and got["warm"]["p50_ms"] == 1
    assert got["cache_speedup_p50"] == 30
    assert got["cold"]["n"] == 5
    assert got["cold"]["p50_ci_low_ms"] <= 30 <= got["cold"]["p50_ci_high_ms"]
    assert got["cold"]["server_p50_ms"] == 5


def test_run_latency_writes_a_summary_and_raw_per_mode_timings(tmp_path):
    recorder = Recorder()
    summary = run_latency(recorder.client, "idx", MODES, POOL, LatencySettings(per_mode=3, warmup=1),
                          BenchmarkSettings(), tmp_path, sleep=NO_SLEEP, clock=StepClock(0.01))
    assert set(summary) == set(MODES)
    on_disk = read_json(tmp_path, "latency.json")
    assert on_disk["per_mode"] == 3 and on_disk["modes"]["rrf"]["cold"]["n"] == 3
    raw = read_latency(tmp_path, "semantic")
    assert len(raw["slice_query_ids"]) == 3 and len(raw["cold"]["wall_ms"]) == 3
