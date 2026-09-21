import json

import pytest

from artifacts import list_modes, read_json, read_runs
from benchmark import (
    BenchmarkSettings,
    IndexIncomplete,
    measure_mode,
    run_benchmark,
    verify_index_complete,
)
from dataset import Query
from fakes import ScriptedSearchClient, StepClock

QUERIES = [Query("q1", "alpha"), Query("q2", "beta")]
QRELS = {"q1": {"a": 1}, "q2": {"b": 1}}
NO_SLEEP = lambda seconds: None  # noqa: E731


def hits(text, request):
    return {"alpha": ["a", "x"], "beta": ["x", "b"]}[text]


def settings(**overrides):
    return BenchmarkSettings(**{"warmup": 0, **overrides})


def test_warmup_queries_are_executed_but_not_recorded():
    client = ScriptedSearchClient(hits)
    runs, _ = measure_mode(client, "idx", "bm25", QUERIES, settings(warmup=2, repeats=1),
                           sleep=NO_SLEEP, clock=StepClock())
    assert len(client.calls) == 2 + len(QUERIES)  # 2 warm-up + 2 measured
    assert [r.query_id for r in runs] == ["q1", "q2"]


def test_latency_is_the_per_query_median_across_repeats():
    client = ScriptedSearchClient(lambda t, r: ["a"])
    # three passes over one query; client-side durations 10, 30, 20 ms -> median 20
    ticks = iter([0.0, 0.010, 1.0, 1.030, 2.0, 2.020])
    runs, _ = measure_mode(client, "idx", "bm25", [Query("q1", "alpha")],
                           settings(repeats=3), sleep=NO_SLEEP, clock=ticks.__next__)
    assert runs[0].wall_ms == pytest.approx(20.0)


def test_the_first_passs_ranking_is_kept_and_instability_is_flagged():
    seen = {"alpha": 0}

    def flaky(text, request):
        if text != "alpha":
            return ["b"]
        seen["alpha"] += 1
        return ["a"] if seen["alpha"] == 1 else ["z"]  # changes on the second pass

    runs, unstable = measure_mode(ScriptedSearchClient(flaky), "idx", "bm25", QUERIES,
                                  settings(repeats=2), sleep=NO_SLEEP, clock=StepClock())
    assert {r.query_id: r.ranked_ids for r in runs} == {"q1": ["a"], "q2": ["b"]}
    assert unstable == ["q1"]


def test_stable_rankings_report_no_instability():
    _, unstable = measure_mode(ScriptedSearchClient(hits), "idx", "bm25", QUERIES,
                               settings(repeats=3), sleep=NO_SLEEP, clock=StepClock())
    assert unstable == []


def test_a_partial_index_is_refused_and_a_complete_one_passes():
    with pytest.raises(IndexIncomplete, match="560 documents, expected 5183"):
        verify_index_complete(ScriptedSearchClient(hits, doc_count=560), "idx", 5183)
    assert verify_index_complete(ScriptedSearchClient(hits, doc_count=5183), "idx", 5183) == 5183


def test_run_benchmark_scores_and_writes_every_artifact(tmp_path):
    client = ScriptedSearchClient(hits, took=7.0)
    summary = run_benchmark(client, "idx", ["bm25", "rrf"], QUERIES, QRELS, settings(),
                            tmp_path, sleep=NO_SLEEP, clock=StepClock(0.010))
    # q1 finds its document at rank 1, q2 at rank 2 -> MRR (1 + 0.5) / 2
    assert summary["bm25"]["mrr@10"] == pytest.approx(0.75)
    assert summary["bm25"]["recall@10"] == 1.0
    assert summary["bm25"]["p50_ms"] == pytest.approx(10.0)
    assert summary["bm25"]["server_p50_ms"] == 7.0
    assert list_modes(tmp_path) == ["bm25", "rrf"]
    assert read_runs(tmp_path, "bm25")["q2"].ranked_ids == ["x", "b"]
    on_disk = read_json(tmp_path, "metrics.json")
    assert on_disk["queries"] == 2 and on_disk["modes"]["bm25"]["mrr@10"] == pytest.approx(0.75)
    json.dumps(on_disk)  # must be serializable as written


def test_settings_reach_the_request_body(tmp_path):
    client = ScriptedSearchClient(hits)
    run_benchmark(client, "idx", ["rrf"], QUERIES, QRELS, settings(k=5, window=30, rank_constant=10),
                  tmp_path, sleep=NO_SLEEP, clock=StepClock())
    body = client.calls[0]
    assert body["size"] == 5
    assert body["retriever"]["rrf"]["rank_window_size"] == 30
    assert body["retriever"]["rrf"]["rank_constant"] == 10


def test_an_unknown_mode_fails_before_any_request_is_sent(tmp_path):
    client = ScriptedSearchClient(hits)
    with pytest.raises(ValueError, match="unknown modes"):
        run_benchmark(client, "idx", ["bm25", "hybrid"], QUERIES, QRELS, settings(), tmp_path)
    assert client.calls == [] and not (tmp_path / "metrics.json").exists()


def test_a_judged_query_missing_from_the_query_list_is_an_error_not_a_skip(tmp_path):
    with pytest.raises(ValueError, match="no run"):
        run_benchmark(ScriptedSearchClient(hits), "idx", ["bm25"], QUERIES[:1], QRELS, settings(),
                      tmp_path, sleep=NO_SLEEP, clock=StepClock())
