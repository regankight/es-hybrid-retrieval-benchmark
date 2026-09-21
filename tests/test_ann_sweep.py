from dataclasses import asdict

import pytest

from ann_sweep import (MODES, SweepSettings, build_ann_body, label, overlap_at_k, render_markdown, run_sweep,
                       summarize)
from artifacts import read_json
from dataset import Query
from fakes import StepClock, api_error

MODEL = ".multilingual-e5-small-elasticsearch"


def knn_of(body):
    query = body["query"] if "query" in body else body["retriever"]["rrf"]["retrievers"][1]["standard"]["query"]
    return query["knn"]


def test_the_dense_body_is_an_explicit_knn_query_with_the_requested_num_candidates():
    body = build_ann_body("dense", "some text", MODEL, num_candidates=200, depth=20)
    knn = knn_of(body)
    assert (knn["k"], knn["num_candidates"], knn["field"]) == (20, 200, "semantic")
    assert knn["query_vector_builder"]["text_embedding"] == {"model_id": MODEL, "model_text": "some text"}
    assert body["size"] == 20


def test_rrf_wraps_the_knn_query_in_a_standard_retriever_next_to_bm25():
    # The knn retriever rejects semantic_text fields, so the knn query rides in a standard retriever.
    body = build_ann_body("rrf", "t", MODEL, num_candidates=50, depth=20, rank_constant=60)
    legs = body["retriever"]["rrf"]["retrievers"]
    assert "multi_match" in legs[0]["standard"]["query"]
    assert knn_of(body)["num_candidates"] == 50
    assert body["retriever"]["rrf"]["rank_window_size"] == 20


def test_filters_gate_the_dense_leg_and_the_keyword_leg():
    f = [{"term": {"x": 1}}]
    body = build_ann_body("rrf", "t", MODEL, num_candidates=50, filters=f)
    legs = body["retriever"]["rrf"]["retrievers"]
    assert all(leg["standard"]["query"]["bool"]["filter"] == f for leg in legs)


def test_num_candidates_below_k_or_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="at least depth"):
        build_ann_body("dense", "t", MODEL, num_candidates=10, depth=20)
    with pytest.raises(ValueError, match="unknown mode"):
        build_ann_body("bm25", "t", MODEL, num_candidates=50)


@pytest.mark.parametrize("kwargs", [
    dict(candidates=(20,)), dict(candidates=(50, 20)), dict(candidates=(20, 20, 50)),
    dict(candidates=(10, 50)), dict(candidates=(20, 20000)), dict(candidates=(20, 50), latency_rounds=0),
    dict(candidates=(20, 50), k=30),
])
def test_invalid_sweep_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SweepSettings(**kwargs)


def test_overlap_is_the_share_of_the_reference_top_k_that_is_returned():
    assert overlap_at_k(["a", "b", "c", "x"], ["a", "b", "c", "d"], 4) == 0.75
    assert overlap_at_k(["a", "b"], ["b", "a"], 2) == 1.0  # order inside the top-k does not matter
    assert overlap_at_k(["z"], [], 3) == 1.0


QUERIES = [Query("q1", "alpha"), Query("q2", "beta")]
QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}
SETTINGS = SweepSettings(candidates=(20, 40), k=10, depth=20, latency_rounds=2)


def hits_for(text, request):
    nc = knn_of(request)["num_candidates"]
    # the small setting misses q2's relevant document, the large one finds it
    return ["d1", "x"] if text == "alpha" else (["d2", "x"] if nc == 40 else ["y", "x"])


class KnnClient:
    """Answers searches whose text lives in a knn query's builder, which the shared fake cannot read."""

    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = [], fail_on_call

    def search(self, **request):
        self.calls.append(request)
        if len(self.calls) == self.fail_on_call:
            raise api_error(503)
        text = knn_of(request)["query_vector_builder"]["text_embedding"]["model_text"]
        return {"took": 3.0, "hits": {"hits": [{"_id": i} for i in hits_for(text, request)]}}


def swept(tmp_path, client=None, settings=SETTINGS):
    client = client or KnnClient()
    timings = run_sweep(client, "idx", QUERIES, MODEL, settings, tmp_path, sleep=lambda s: None, clock=StepClock())
    return client, timings


def combos_count():
    return len(MODES) * len(SETTINGS.candidates)


def test_the_sweep_ranks_every_query_at_every_setting_then_times_the_configured_rounds(tmp_path):
    client, timings = swept(tmp_path)
    assert len(client.calls) == combos_count() * len(QUERIES) * (1 + SETTINGS.latency_rounds)
    assert all(len(t.wall_ms) == len(QUERIES) * SETTINGS.latency_rounds for t in timings.values())
    assert set(read_json(tmp_path, "latency.json")["settings"]) == {label(m, nc) for m in MODES for nc in (20, 40)}


def test_no_setting_always_goes_first_in_the_timed_rounds(tmp_path):
    client, _ = swept(tmp_path)
    timed = client.calls[combos_count() * len(QUERIES):]
    firsts = {knn_of(timed[i])["num_candidates"] for i in range(0, len(timed), combos_count())}
    assert firsts == {20, 40}


def test_summary_measures_recall_against_the_reference_and_paired_quality(tmp_path):
    swept(tmp_path)
    summary = summarize(tmp_path, QRELS, asdict(SETTINGS))
    low, high = summary["modes"]["dense"]
    assert (low["num_candidates"], high["num_candidates"]) == (20, 40)
    assert high["ann_recall"] == 1.0 and high["queries_not_identical_to_reference"] == 0
    assert low["queries_not_identical_to_reference"] == 1  # q2 differs from the reference
    assert low["ndcg_vs_reference"]["estimate"] < 0 and high["ndcg_vs_reference"]["estimate"] == 0
    assert low["ndcg"]["estimate"] == pytest.approx(0.5)


def test_a_retry_in_the_timed_rounds_is_counted_and_one_in_the_untimed_pass_is_not(tmp_path):
    one_round = SweepSettings(candidates=(20, 40), latency_rounds=1)
    _, untimed = swept(tmp_path / "a", KnnClient(fail_on_call=1), one_round)
    assert sum(t.retried for t in untimed.values()) == 0
    _, timed = swept(tmp_path / "b", KnnClient(fail_on_call=combos_count() * len(QUERIES) + 1), one_round)
    assert sum(t.retried for t in timed.values()) == 1


def test_the_report_names_the_reference_and_the_comparison_run(tmp_path):
    swept(tmp_path)
    manifest = {"tag": "t", "index": "idx", "index_documents": 9, "queries": 2, "started_utc": "a",
                "finished_utc": "b", "code": {"commit": "abcdef1234567890"}, "settings": asdict(SETTINGS),
                "compare": None}
    text = render_markdown(manifest, summarize(tmp_path, QRELS, manifest["settings"]))
    assert "`num_candidates` = 40" in text and "not a proven-exact search" in text
    assert "## Dense (e5, knn)" in text and "## RRF hybrid" in text
