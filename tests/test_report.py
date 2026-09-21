import math

import pytest

from artifacts import write_json, write_rankings
from dataset import Document, Query
from report import (RERANK_MODE, breakdown, build_report, compare, describe, fusion_categories,
                    pick_examples, pool_ceiling, rerank_examples)

# Six queries, one relevant document each, built so every category has exactly one case.
#   q1 keyword wins   q2 semantic wins   q3 fusion hurts   q4 fusion helps   q5 all fail   q6 all tie
QRELS = {f"q{i}": {f"d{i}": 1} for i in range(1, 7)}
QUERIES = [Query(f"q{i}", f"query text {i}") for i in range(1, 7)]
CORPUS = [Document(f"d{i}", f"Title {i}", "body") for i in range(1, 11)]
FILLER = ["d7", "d8", "d9"]

RANKINGS = {
    "bm25":     {"q1": ["d1"] + FILLER, "q2": FILLER, "q3": ["d3"] + FILLER, "q4": ["d7", "d4", "d8"],
                 "q5": FILLER, "q6": ["d6"] + FILLER},
    "semantic": {"q1": FILLER, "q2": ["d2"] + FILLER, "q3": ["d3"] + FILLER, "q4": ["d8", "d4", "d9"],
                 "q5": FILLER, "q6": ["d6"] + FILLER},
    "rrf":      {"q1": ["d1"] + FILLER, "q2": ["d2"] + FILLER, "q3": FILLER, "q4": ["d4"] + FILLER,
                 "q5": FILLER, "q6": ["d6"] + FILLER},
}


def ndcg_of(rank):  # a single relevant document at 1-based `rank`
    return 1 / math.log2(rank + 1)


def scores():
    from metrics import per_query_scores
    return {m: per_query_scores(RANKINGS[m], QRELS, 10)["ndcg@10"] for m in RANKINGS}


def test_compare_counts_wins_losses_and_ties_and_signs_the_interval():
    base = {f"q{i}": 0.5 for i in range(20)}
    cand = {q: 0.5 + (0.2 if i < 12 else -0.2 if i < 15 else 0.0) for i, q in enumerate(base)}
    got = compare(base, cand, "bm25", "rrf", resamples=2000, seed=1)
    assert (got.wins, got.losses, got.ties) == (12, 3, 5)
    assert got.interval.estimate > 0 and got.baseline == "bm25" and got.candidate == "rrf"


def test_a_difference_below_the_tie_threshold_is_a_tie():
    got = compare({"a": 0.5}, {"a": 0.5 + 1e-12}, "x", "y", resamples=100)
    assert (got.wins, got.losses, got.ties) == (0, 0, 1)


def test_every_category_finds_exactly_its_query():
    picked = pick_examples(scores())
    assert picked == {"keyword_wins": ["q1"], "semantic_wins": ["q2"], "fusion_hurts": ["q3"],
                      "fusion_helps": ["q4"], "all_fail": ["q5"]}


def test_examples_are_ordered_by_margin_then_id_and_capped():
    s = {"bm25": {"a": 1.0, "b": 0.6, "c": 0.9, "d": 0.9}, "semantic": {q: 0.0 for q in "abcd"},
         "rrf": {q: 1.0 for q in "abcd"}}
    got = pick_examples(s, per_category=3)["keyword_wins"]
    assert got == ["a", "c", "d"]  # 1.0, then the 0.9 tie broken by id, and "b" (0.6) is cut by the cap


def test_describe_reports_where_each_mode_ranked_the_first_relevant_document():
    ex = describe(QUERIES[3], QRELS["q4"], RANKINGS, scores(), {d.id: d.title for d in CORPUS})
    assert ex["relevant_titles"] == ["Title 4"]
    assert ex["modes"]["bm25"]["first_relevant_rank"] == 2
    assert ex["modes"]["rrf"]["first_relevant_rank"] == 1
    assert ex["modes"]["rrf"]["ndcg"] == pytest.approx(1.0)
    assert ex["modes"]["bm25"]["top3"] == ["Title 7", "Title 4", "Title 8"]
    miss = describe(QUERIES[4], QRELS["q5"], RANKINGS, scores(), {})
    assert miss["modes"]["bm25"]["first_relevant_rank"] is None  # never retrieved


def _write_run(run_dir):
    for mode, rankings in RANKINGS.items():
        write_rankings(run_dir, mode, rankings)
    node = {"allocated_processors": 10, "memory_total_bytes": 8 * 2**30, "os": "Linux", "arch": "aarch64"}
    write_json(run_dir, "manifest.json", {
        "tag": "test-run", "started_utc": "2026-01-01T00:00:00+00:00", "finished_utc": "2026-01-01T00:10:00+00:00",
        "note": "synthetic", "settings": {"k": 10},
        "dataset": {"corpus_documents": len(CORPUS)}, "code": {"commit": "abcdef1234567890", "dirty": False},
        "environment": {"elasticsearch_server": "9.5.2", "elasticsearch_client": "9.5.1",
                        "license": {"type": "trial"}, "node": node},
    })
    cold = {"p50_ms": 85.0, "p50_ci_low_ms": 80.0, "p50_ci_high_ms": 90.0, "p95_ms": 140.0}
    write_json(run_dir, "latency.json", {"per_mode": 200, "order": "interleaved", "modes": {
        m: {"cold": cold, "warm": {"p50_ms": 2.0}, "cache_speedup_p50": 42.5} for m in RANKINGS}})


def test_the_report_contains_every_section_and_hand_checkable_numbers(tmp_path):
    _write_run(tmp_path)
    text = build_report(tmp_path, CORPUS, QUERIES, QRELS)
    for heading in ("# Report: test-run", "## Retrieval quality (6 queries", "## Paired comparisons",
                    "## Latency", "## Failure analysis", "### Where every mode failed"):
        assert heading in text
    # BM25 finds d1, d3, d6 at rank 1 and d4 at rank 2: mean nDCG@10 = (3 + ndcg_of(2)) / 6
    expected = (3 + ndcg_of(2)) / 6
    assert f"{expected:.4f}" in text
    assert "`abcdef1234`" in text and "working tree clean" in text
    assert "**Query q5:** query text 5" in text


def test_the_report_is_reproducible(tmp_path):
    _write_run(tmp_path)
    assert build_report(tmp_path, CORPUS, QUERIES, QRELS) == build_report(tmp_path, CORPUS, QUERIES, QRELS)


def test_pool_ceiling_separates_unfixable_from_rescuable_queries():
    rankings = {"q1": ["x", "d1", "y", "z"],  # answer at rank 2: in the pool, outside a cutoff of 1
                "q2": ["d2", "x", "y", "z"],  # answer already at rank 1
                "q3": ["x", "y", "z", "w"]}   # answer nowhere: no reranker can fix this
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    got = pool_ceiling(rankings, qrels, pool=4, k=1)
    assert got == {"recall_at_k": pytest.approx(1 / 3), "recall_at_pool": pytest.approx(2 / 3),
                   "none_in_pool": 1, "rescuable": 1}


def test_fusion_categories_partition_every_query_exactly_once():
    cats = fusion_categories(scores())
    everyone = [q for qs in cats.values() for q in qs]
    assert sorted(everyone) == sorted(QRELS) and len(everyone) == len(set(everyone))
    assert cats["fusion hurt (RRF below the better single mode)"] == ["q3"]
    assert cats["fusion helped (RRF above both)"] == ["q4"]


def test_breakdown_reports_mean_before_and_after_and_skips_empty_categories():
    before, after = {"a": 0.2, "b": 0.4, "c": 1.0}, {"a": 0.6, "b": 0.8, "c": 0.5}
    rows = breakdown(before, after, {"first": ["a", "b"], "second": ["c"], "empty": []})
    assert rows == [("first", 2, pytest.approx(0.3), pytest.approx(0.7)), ("second", 1, 1.0, 0.5)]


def test_rerank_examples_are_the_largest_gains_and_losses_with_ties_broken_by_id():
    before = {"a": 0.0, "b": 0.5, "c": 0.5, "d": 1.0}
    after = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.2}
    got = rerank_examples(before, after, per_category=2)
    assert got["rerank_helps"] == ["a", "b"]  # +1.0, then the +0.5 tie broken by id
    assert got["rerank_hurts"] == ["d"]


def _write_rerank_run(rerank_dir):
    reranked = {**{q: r[:3] for q, r in RANKINGS["rrf"].items()}, "q3": ["d3", "d7", "d8"]}  # the rerank fixes q3
    write_rankings(rerank_dir, RERANK_MODE, reranked)
    write_json(rerank_dir, "metrics.json", {})
    write_json(rerank_dir, "latency.json", {"p50_ms": 357.0, "p50_ci_low_ms": 356.0, "p50_ci_high_ms": 358.0, "p95_ms": 373.0})
    write_json(rerank_dir, "manifest.json", {
        "candidates": 3, "source_rankings_sha256": "f" * 64, "code": {"commit": "1234567890abcdef", "dirty": False},
        "model": {"repo": "org/model", "revision": "abcdef1234567890", "license": "apache-2.0",
                  "parameters": 22_700_000, "torch_threads": 4},
    })


def test_the_report_gains_a_reranking_section_when_given_a_rerank_run(tmp_path):
    run, rerank_dir = tmp_path / "run", tmp_path / "rerank"
    _write_run(run)
    _write_rerank_run(rerank_dir)
    plain = build_report(run, CORPUS, QUERIES, QRELS)
    text = build_report(run, CORPUS, QUERIES, QRELS, rerank_dir)
    body = lambda t: t.split("## Retrieval quality", 1)[1]  # noqa: E731
    assert "## Reranking" not in plain and body(text).startswith(body(plain).rstrip("\n"))  # purely additive
    for needle in ("## Reranking (offline cross-encoder", "`org/model`", "`abcdef1234`", "apache-2.0", "22.7M parameters",
                   "RRF + local rerank vs RRF hybrid", "Who it helps and who it hurts", "Headroom.",
                   "fusion hurt (RRF below the better single mode)", "Where reranking improved most",
                   "median 357 ms per query [356, 358]", "about 119 ms per candidate",
                   "`python report.py artifacts/test-run --rerank artifacts/rerank`"):
        assert needle in text, needle
    assert text == build_report(run, CORPUS, QUERIES, QRELS, rerank_dir)  # reproducible
