import math

import pytest

from metrics import (
    evaluate,
    latency_summary,
    ndcg_at_k,
    per_query_scores,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_ndcg_matches_a_hand_computed_value():
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.5 ; ideal = 1/log2(2) + 1/log2(3) = 1.6309
    got = ndcg_at_k(["d1", "d2", "d3"], {"d1": 1, "d3": 1}, k=3)
    assert got == pytest.approx(0.9197, abs=1e-4)


def test_ndcg_uses_graded_relevance():
    # ranked [b(1), a(2)]: DCG = 1 + 2/log2(3) ; ideal [a, b] = 2 + 1/log2(3)
    got = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=2)
    assert got == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))
    assert got == pytest.approx(0.8597, abs=1e-4)


def test_ndcg_perfect_ranking_is_one_and_miss_is_zero():
    judgments = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], judgments, k=10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "z"], judgments, k=10) == 0.0


def test_ndcg_applies_the_cutoff():
    assert ndcg_at_k(["x", "y", "z", "a"], {"a": 1}, k=3) == 0.0
    assert ndcg_at_k(["x", "y", "z", "a"], {"a": 1}, k=4) > 0.0


def test_ndcg_ignores_negative_grades_and_unjudged_documents():
    assert ndcg_at_k(["bad", "unjudged", "a"], {"bad": -1, "a": 1}, k=3) == pytest.approx(0.5)


def test_recall_counts_relevant_documents_found_in_the_top_k():
    judgments = {"a": 1, "b": 1, "c": 1}
    assert recall_at_k(["a", "x", "b", "y"], judgments, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "x", "b", "y"], judgments, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["a"], {}, k=10) == 0.0


def test_reciprocal_rank_is_one_over_the_first_relevant_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, k=10) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"a": 1}, k=10) == 1.0
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, k=2) == 0.0  # outside the cutoff
    assert reciprocal_rank(["x"], {"a": 1}, k=10) == 0.0


def test_evaluate_averages_over_every_judged_query():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    runs = {"q1": ["a"], "q2": ["x", "b"]}
    got = evaluate(runs, qrels, k=10)
    assert got["mrr@10"] == pytest.approx((1.0 + 0.5) / 2)
    assert got["recall@10"] == pytest.approx(1.0)
    assert got["ndcg@10"] == pytest.approx((1.0 + 1 / math.log2(3)) / 2)


def test_a_judged_query_with_no_run_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="no run"):
        evaluate({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}})


def test_per_query_scores_stay_available_for_paired_comparison():
    scores = per_query_scores({"q1": ["a"], "q2": ["x"]}, {"q1": {"a": 1}, "q2": {"b": 1}}, k=5)
    assert scores["ndcg@5"] == {"q1": 1.0, "q2": 0.0}


def test_percentile_interpolates_like_numpy():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([7], 95) == 7
    with pytest.raises(ValueError):
        percentile([], 50)


def test_latency_summary():
    got = latency_summary([10, 20, 30, 40])
    assert got == {"p50_ms": 25.0, "p95_ms": pytest.approx(38.5), "mean_ms": 25.0}
