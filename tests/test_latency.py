import json
from datetime import datetime, timezone

import pytest

from artifacts import read_json, read_latency
from benchmark import BenchmarkSettings
from dataset import Query
from fakes import ScriptedSearchClient, StepClock, api_error
from latency import (
    LatencySettings,
    ModeLatency,
    build_latency_plan,
    cache_conflicts,
    load_prior_runs,
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
    return "rrf_rerank" if "text_similarity_reranker" in request["retriever"] else "rrf"


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


def test_retried_queries_are_counted_instead_of_silently_absorbed():
    recorder = Recorder()
    recorder.client.errors = [api_error(502)]  # the very first measured request fails once
    latency = measure(recorder, per_mode=3, warmup=0)
    assert latency["bm25"].retried == 1
    assert latency["semantic"].retried == 0 and latency["rrf"].retried == 0
    assert summarize_latency(latency["bm25"])["retried_queries"] == 1
    assert summarize_latency(latency["rrf"])["retried_queries"] == 0


def _ml(cold, warm):
    n = len(cold)
    return ModeLatency(query_ids=[f"q{i}" for i in range(n)], cold_wall_ms=cold, cold_took_ms=[1.0] * n,
                       warm_wall_ms=warm, warm_took_ms=[1.0] * n)


def test_a_cold_pass_half_full_of_cached_speed_queries_is_flagged_as_contaminated():
    # the exact shape release-1 had: half the "cold" queries were already in the cache
    got = summarize_latency(_ml([2.0] * 10 + [90.0] * 10, [2.0] * 20), check_cache=True)
    assert got["cached_like_cold_queries"] == 10
    assert got["cache_contaminated"] is True


def test_a_genuinely_cold_pass_is_not_flagged():
    got = summarize_latency(_ml([90.0] * 20, [2.0] * 20), check_cache=True)
    assert got["cached_like_cold_queries"] == 0 and got["cache_contaminated"] is False


def test_a_mode_that_never_embeds_is_not_flagged_for_looking_cached():
    # BM25 is legitimately as fast cold as cached; only embedding modes can be contaminated
    got = summarize_latency(_ml([3.0] * 20, [2.0] * 20), check_cache=False)
    assert got["cached_like_cold_queries"] == 20 and got["cache_contaminated"] is False


def test_a_stray_cached_query_within_tolerance_is_not_a_false_alarm():
    got = summarize_latency(_ml([2.0] + [90.0] * 39, [2.0] * 40), check_cache=True)  # 2.5% < 5%
    assert got["cached_like_cold_queries"] == 1 and got["cache_contaminated"] is False


def test_run_latency_applies_the_contamination_check_to_embedding_modes_only(tmp_path):
    # Every query "costs" 2 ms cold and cached alike: what a fully contaminated cold pass looks like.
    summary = run_latency(Recorder().client, "idx", MODES, POOL, LatencySettings(per_mode=10, warmup=1),
                          BenchmarkSettings(), tmp_path, sleep=NO_SLEEP, clock=StepClock(0.002))
    assert summary["bm25"]["cache_contaminated"] is False  # BM25 embeds nothing
    assert summary["semantic"]["cache_contaminated"] is True
    assert summary["rrf"]["cache_contaminated"] is True
    assert read_json(tmp_path, "latency.json")["modes"]["semantic"]["cache_contaminated"] is True


def test_the_cache_check_is_not_applied_to_the_rerank_mode_whose_cold_and_cached_passes_look_alike():
    # An Elasticsearch rerank spends seconds on uncached inference, so a valid cold pass is no
    # slower than the cached one by more than the ~85 ms embedding. That must not read as contamination.
    slow = _ml([5200.0 + i for i in range(20)], [5115.0 + i for i in range(20)])
    assert summarize_latency(slow, check_cache=True)["cache_contaminated"] is True   # the check would misfire...
    assert summarize_latency(slow, check_cache=False)["cache_contaminated"] is False  # ...so it is switched off


def test_run_latency_checks_the_cache_for_semantic_and_rrf_only(tmp_path):
    modes = ["bm25", "semantic", "rrf", "rrf_rerank"]
    summary = run_latency(Recorder().client, "idx", modes, POOL, LatencySettings(per_mode=9, warmup=1),
                          BenchmarkSettings(), tmp_path, sleep=NO_SLEEP, clock=StepClock(0.002))
    assert {m: summary[m]["cache_check"] for m in modes} == {"bm25": False, "semantic": True, "rrf": True, "rrf_rerank": False}
    assert summary["rrf_rerank"]["cache_contaminated"] is False  # identical timings, but not checked
    assert summary["semantic"]["cache_contaminated"] is True and summary["rrf"]["cache_contaminated"] is True


def test_a_zero_millisecond_cached_median_gives_no_speedup_instead_of_a_crash():
    got = summarize_latency(_ml([3.0, 3.0, 3.0], [0.0, 0.0, 0.0]))
    assert got["cache_speedup_p50"] is None
    json.dumps(got)  # must still serialize into latency.json


def _prior_run(root, tag, finished, ids, *, with_latency=True):
    run = root / tag
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"finished_utc": finished}))
    if with_latency:
        (run / "latency").mkdir()
        (run / "latency" / "semantic.json").write_text(json.dumps({"slice_query_ids": ids}))


def test_prior_runs_are_read_from_disk_skipping_the_current_tag_and_runs_without_latency(tmp_path):
    _prior_run(tmp_path, "release-1", "2026-09-21T16:00:00+00:00", ["h1", "h2"])
    _prior_run(tmp_path, "quality-only", "2026-09-21T16:30:00+00:00", ["h3"], with_latency=False)
    _prior_run(tmp_path, "current", "2026-09-21T16:40:00+00:00", ["h4"])
    runs = load_prior_runs(tmp_path, exclude_tag="current")
    assert [(r.tag, r.query_ids) for r in runs] == [("release-1", frozenset({"h1", "h2"}))]
    assert runs[0].finished == datetime(2026, 9, 21, 16, 0, tzinfo=timezone.utc)


def test_a_restart_after_the_earlier_run_clears_the_conflict_and_no_restart_keeps_it(tmp_path):
    _prior_run(tmp_path, "release-1", "2026-09-21T16:00:00+00:00", ["h1", "h2"])
    prior = load_prior_runs(tmp_path, exclude_tag="new")
    before = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)   # ES started BEFORE the earlier run: cache is warm
    after = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)    # ES restarted AFTER it: cache is empty
    assert cache_conflicts(prior, {"h2", "h9"}, before) == ["release-1"]
    assert cache_conflicts(prior, {"h2", "h9"}, after) == []


def test_no_overlap_means_no_conflict_even_without_a_restart(tmp_path):
    _prior_run(tmp_path, "release-1", "2026-09-21T16:00:00+00:00", ["h1", "h2"])
    before = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)
    assert cache_conflicts(load_prior_runs(tmp_path, "new"), {"h8", "h9"}, before) == []
