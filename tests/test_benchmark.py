import json

import pytest

from artifacts import list_modes, read_json, read_rankings
from benchmark import (
    BenchmarkSettings,
    IndexIncomplete,
    measure_rankings,
    run_benchmark,
    verify_index_complete,
)
from dataset import Query
from fakes import ScriptedSearchClient, StepClock, api_error

QUERIES = [Query("q1", "alpha"), Query("q2", "beta")]
QRELS = {"q1": {"a": 1}, "q2": {"b": 1}}
NO_SLEEP = lambda seconds: None  # noqa: E731


def hits(text, request):
    return {"alpha": ["a", "x"], "beta": ["x", "b"]}[text]


def settings(**overrides):
    return BenchmarkSettings(**{"warmup": 0, "stability_sample": 0, **overrides})


def rank(client, queries=QUERIES, **overrides):
    return measure_rankings(client, "idx", "bm25", queries, settings(**overrides),
                            sleep=NO_SLEEP, clock=StepClock())


def test_warmup_queries_are_executed_but_not_recorded():
    client = ScriptedSearchClient(hits)
    rankings, _ = rank(client, warmup=2)
    assert len(client.calls) == 2 + len(QUERIES)
    assert set(rankings) == {"q1", "q2"}


def test_the_stability_check_reruns_only_the_sampled_queries():
    client = ScriptedSearchClient(hits)
    rank(client, stability_sample=1)
    assert len(client.calls) == len(QUERIES) + 1


def test_rankings_that_change_between_passes_are_flagged_and_the_first_pass_is_kept():
    seen = {"alpha": 0}

    def flaky(text, request):
        if text != "alpha":
            return ["b"]
        seen["alpha"] += 1
        return ["a"] if seen["alpha"] == 1 else ["z"]  # changes on the re-run

    rankings, unstable = rank(ScriptedSearchClient(flaky), stability_sample=2)
    assert rankings == {"q1": ["a"], "q2": ["b"]}
    assert unstable == ["q1"]


def test_stable_rankings_report_no_instability():
    _, unstable = rank(ScriptedSearchClient(hits), stability_sample=2)
    assert unstable == []


def test_a_partial_index_is_refused_and_a_complete_one_passes():
    with pytest.raises(IndexIncomplete, match="560 documents, expected 5183"):
        verify_index_complete(ScriptedSearchClient(hits, doc_count=560), "idx", 5183)
    assert verify_index_complete(ScriptedSearchClient(hits, doc_count=5183), "idx", 5183) == 5183


def test_run_benchmark_scores_writes_rankings_and_records_no_latency(tmp_path):
    client = ScriptedSearchClient(hits, took=7.0)
    summary = run_benchmark(client, "idx", ["bm25", "rrf"], QUERIES, QRELS, settings(),
                            tmp_path, sleep=NO_SLEEP, clock=StepClock())
    # q1 finds its document at rank 1, q2 at rank 2 -> MRR (1 + 0.5) / 2
    assert summary["bm25"]["mrr@10"] == pytest.approx(0.75)
    assert summary["bm25"]["recall@10"] == 1.0
    assert list_modes(tmp_path) == ["bm25", "rrf"]
    assert read_rankings(tmp_path, "bm25")["q2"] == ["x", "b"]
    on_disk = read_json(tmp_path, "metrics.json")
    assert on_disk["queries"] == 2 and on_disk["modes"]["bm25"]["mrr@10"] == pytest.approx(0.75)
    json.dumps(on_disk)
    # Regression guard for the bug that motivated latency.py: timings taken while scoring
    # quality are flattered by Elasticsearch's embedding cache and must never be recorded.
    assert not any("ms" in key for key in on_disk["modes"]["bm25"])


def test_settings_reach_the_request_body(tmp_path):
    client = ScriptedSearchClient(hits)
    run_benchmark(client, "idx", ["rrf"], QUERIES, QRELS, settings(k=5, window=30, rank_constant=10),
                  tmp_path, sleep=NO_SLEEP, clock=StepClock())
    body = client.calls[0]
    assert body["size"] == 20  # the default depth: rankings are saved deeper than the cutoff k
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


def test_the_retry_budget_setting_reaches_the_query():
    client = ScriptedSearchClient(hits)
    client.errors = [api_error(408)] * 3
    with pytest.raises(Exception) as excinfo:
        rank(client, attempts=2)  # a budget of 2 must give up on the second failure
    assert excinfo.value.status_code == 408
    assert len(client.calls) == 2


def test_rankings_are_saved_to_depth_but_metrics_are_still_cut_at_k(tmp_path):
    deep = ["x"] * 14 + ["a"]  # the relevant document sits at rank 15, beyond k=10
    client = ScriptedSearchClient(lambda text, request: deep if text == "alpha" else ["b"])
    summary = run_benchmark(client, "idx", ["bm25"], QUERIES, QRELS, settings(k=10, depth=20),
                            tmp_path, sleep=NO_SLEEP, clock=StepClock())
    assert client.calls[0]["size"] == 20
    assert read_rankings(tmp_path, "bm25")["q1"] == deep  # candidates survive for later stages
    assert summary["bm25"]["recall@10"] == pytest.approx(0.5)  # q1's hit at rank 15 does not count


def test_depth_must_sit_between_k_and_window():
    with pytest.raises(ValueError, match="depth"):
        BenchmarkSettings(k=10, depth=5, window=20)
    with pytest.raises(ValueError, match="depth"):
        BenchmarkSettings(k=10, depth=30, window=20)
    BenchmarkSettings(k=10, depth=10, window=10)
