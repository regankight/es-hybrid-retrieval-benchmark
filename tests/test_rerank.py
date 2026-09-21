import pytest

from artifacts import read_json, read_rankings, write_rankings
from dataset import Document, Query
from fakes import StepClock
from indexing import document_source
from rerank_offline import (
    MODE,
    SourceError,
    passage,
    rerank,
    rerank_queries,
    run_rerank,
    sha256_of,
    verify_source,
)

DOCS = [Document(f"d{i}", f"Title {i}", f"body {i}") for i in range(1, 7)]
TEXTS = {d.id: passage(d) for d in DOCS}
QUERIES = [Query("q1", "alpha"), Query("q2", "beta")]
QRELS = {"q1": {"d3": 1}, "q2": {"d5": 1}}
CANDIDATES = {"q1": ["d1", "d2", "d3", "d4"], "q2": ["d5", "d1", "d2", "d6"]}  # RRF put d3 third, d5 first


def scorer(prefers):
    """A fake cross-encoder that gives 1.0 to the passage of the doc it 'prefers' per query."""
    def score(query, passages):
        return [1.0 if TEXTS[prefers[query]] == p else 0.0 for p in passages]
    return score


def test_passage_matches_the_text_the_dense_model_was_indexed_with():
    for doc in (Document("1", "A title", "the abstract"), Document("2", "", "no title here")):
        assert passage(doc) == document_source(doc)["semantic"]


def test_rerank_orders_best_first_and_preserves_every_candidate():
    assert rerank(["a", "b", "c"], [0.1, 0.9, 0.5]) == ["b", "c", "a"]


def test_ties_keep_the_incoming_order_so_a_useless_reranker_leaves_rrf_alone():
    assert rerank(["a", "b", "c", "d"], [0.0, 0.0, 0.0, 0.0]) == ["a", "b", "c", "d"]
    assert rerank(["a", "b", "c"], [0.5, 1.0, 0.5]) == ["b", "a", "c"]


def test_reranking_can_move_the_relevant_document_up_and_leave_a_correct_one_alone():
    rankings, _, _ = rerank_queries(QUERIES, CANDIDATES, TEXTS, scorer({"alpha": "d3", "beta": "d5"}),
                                    warmup=0, stability_sample=0)
    assert rankings["q1"][0] == "d3"  # was third
    assert rankings["q2"][0] == "d5"  # was already first
    assert sorted(rankings["q1"]) == sorted(CANDIDATES["q1"])  # nothing lost or invented


def test_warmup_is_scored_but_discarded_and_stability_rescoring_is_counted():
    calls = []

    def counting(query, passages):
        calls.append(query)
        return [0.0] * len(passages)

    rerank_queries(QUERIES, CANDIDATES, TEXTS, counting, warmup=1, stability_sample=1)
    assert calls == ["alpha"] + ["alpha", "beta"] + ["alpha"]  # warm-up, measured pass, re-score of the sample


def test_a_reranker_whose_output_changes_between_passes_is_flagged():
    state = {"n": 0}

    def flaky(query, passages):
        if query == "alpha":
            state["n"] += 1
            return [1.0 if (state["n"] == 1 and i == 0) or (state["n"] > 1 and i == 1) else 0.0
                    for i in range(len(passages))]
        return [0.0] * len(passages)

    _, _, unstable = rerank_queries(QUERIES, CANDIDATES, TEXTS, flaky, warmup=0, stability_sample=2)
    assert unstable == ["q1"]


def test_per_query_time_is_the_clock_around_the_scoring_only():
    _, millis, _ = rerank_queries(QUERIES, CANDIDATES, TEXTS, scorer({"alpha": "d1", "beta": "d1"}),
                                  warmup=0, stability_sample=0, clock=StepClock(0.05))
    assert millis == pytest.approx({"q1": 50.0, "q2": 50.0})


def test_a_candidate_without_document_text_is_an_error_not_a_silent_skip():
    with pytest.raises(SourceError, match="no document text"):
        rerank_queries(QUERIES[:1], {"q1": ["d1", "ghost"]}, TEXTS, scorer({"alpha": "d1"}), warmup=0)


def test_a_source_run_that_is_too_shallow_or_incomplete_is_refused():
    with pytest.raises(SourceError, match="fewer than 4"):
        verify_source({"q1": ["a", "b"], "q2": ["a"] * 4}, QUERIES, candidates=4)
    with pytest.raises(SourceError, match="no ranking"):
        verify_source({"q1": ["a"] * 4}, QUERIES, candidates=4)
    verify_source({"q1": ["a"] * 4, "q2": ["a"] * 4}, QUERIES, candidates=4)


def test_run_rerank_writes_scored_artifacts_tied_to_the_source(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    write_rankings(source, "rrf", CANDIDATES)
    result = run_rerank(source, out, QUERIES, QRELS, DOCS, scorer({"alpha": "d3", "beta": "d5"}),
                        candidates=4, k=10, warmup=0, stability_sample=0,
                        provenance={"tag": "t"}, clock=StepClock(0.02))
    # RRF had q1's answer at rank 3 (nDCG 0.5) and q2's at rank 1 (1.0); the rerank fixes q1
    assert result["rrf"]["ndcg@10"] == pytest.approx((0.5 + 1.0) / 2)
    assert result["quality"]["ndcg@10"] == pytest.approx(1.0)
    assert read_rankings(out, MODE)["q1"][0] == "d3"
    metrics, manifest, latency = (read_json(out, n) for n in ("metrics.json", "manifest.json", "latency.json"))
    assert metrics["modes"][MODE]["ndcg@10"] == pytest.approx(1.0)
    assert manifest["source_rankings_sha256"] == sha256_of(source / "rankings" / "rrf.json")
    assert manifest["tag"] == "t" and manifest["candidates"] == 4
    assert latency["n"] == 2 and latency["p50_ms"] == pytest.approx(20.0)
    assert set(latency["per_query_ms"]) == {"q1", "q2"}


def test_the_recorded_source_hash_changes_if_the_candidates_change(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_rankings(a, "rrf", CANDIDATES)
    write_rankings(b, "rrf", {**CANDIDATES, "q1": ["d4", "d3", "d2", "d1"]})
    assert sha256_of(a / "rankings" / "rrf.json") != sha256_of(b / "rankings" / "rrf.json")


def test_only_the_top_candidates_are_reranked_and_the_baseline_uses_the_same_pool(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    write_rankings(source, "rrf", {"q1": ["d1", "d2", "d3", "d4", "d5", "d6"],
                                   "q2": ["d5", "d1", "d2", "d3", "d4", "d6"]})
    qrels = {"q1": {"d5": 1}, "q2": {"d5": 1}}  # q1's answer sits at rank 5, outside a pool of 4
    result = run_rerank(source, out, QUERIES, qrels, DOCS, scorer({"alpha": "d1", "beta": "d5"}),
                        candidates=4, warmup=0, stability_sample=0)
    ranked = read_rankings(out, MODE)
    assert all(len(r) == 4 for r in ranked.values()) and "d5" not in ranked["q1"]
    assert result["rrf"]["ndcg@10"] == pytest.approx(0.5)  # q1 scores 0 in the pool, q2 scores 1.0
