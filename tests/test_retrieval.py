import pytest

from dataset import Query
from fakes import FakeSearchClient, api_error
from retrieval import DEFAULT_RERANK_INFERENCE_ID, MODES, build_search_body, run_query

BM25 = {"multi_match": {"query": "q", "fields": ["title", "text"]}}
SEMANTIC = {"semantic": {"field": "semantic", "query": "q"}}


def test_bm25_body():
    assert build_search_body("bm25", "q") == {"size": 10, "query": BM25}


def test_semantic_body():
    assert build_search_body("semantic", "q") == {"size": 10, "query": SEMANTIC}


def test_rrf_body_fuses_bm25_and_semantic_with_the_default_constants():
    assert build_search_body("rrf", "q") == {
        "size": 10,
        "retriever": {
            "rrf": {
                "retrievers": [{"standard": {"query": BM25}}, {"standard": {"query": SEMANTIC}}],
                "rank_window_size": 20,
                "rank_constant": 60,
            }
        },
    }


def test_rerank_wraps_exactly_the_rrf_retriever():
    rrf = build_search_body("rrf", "q")["retriever"]
    reranked = build_search_body("rrf_rerank", "q")["retriever"]["text_similarity_reranker"]
    assert reranked["retriever"] == rrf  # rerank only re-orders what plain RRF ranked
    assert reranked["field"] == "semantic"
    assert reranked["inference_id"] == DEFAULT_RERANK_INFERENCE_ID
    assert reranked["inference_text"] == "q"


def test_window_is_shared_by_rrf_and_the_reranker():
    body = build_search_body("rrf_rerank", "q", window=30, rank_constant=10)
    reranker = body["retriever"]["text_similarity_reranker"]
    assert reranker["rank_window_size"] == 30
    assert reranker["retriever"]["rrf"]["rank_window_size"] == 30
    assert reranker["retriever"]["rrf"]["rank_constant"] == 10


def test_rerank_options_pass_through():
    body = build_search_body("rrf_rerank", "q", rerank_inference_id="my-reranker", rerank_field="text")
    reranker = body["retriever"]["text_similarity_reranker"]
    assert (reranker["inference_id"], reranker["field"]) == ("my-reranker", "text")


def test_filters_gate_every_retriever_and_never_appear_without_being_asked():
    flt = {"term": {"year": 2025}}
    assert "bool" not in str(build_search_body("rrf", "q"))
    assert build_search_body("bm25", "q", filters=[flt])["query"] == {
        "bool": {"must": [BM25], "filter": [flt]}
    }
    retrievers = build_search_body("rrf", "q", filters=[flt])["retriever"]["rrf"]["retrievers"]
    assert retrievers[0]["standard"]["query"] == {"bool": {"must": [BM25], "filter": [flt]}}
    assert retrievers[1]["standard"]["query"] == {"bool": {"must": [SEMANTIC], "filter": [flt]}}


def test_invalid_arguments_are_rejected_before_any_request():
    with pytest.raises(ValueError, match="unknown mode"):
        build_search_body("hybrid", "q")
    with pytest.raises(ValueError, match="size"):
        build_search_body("bm25", "q", size=0)
    for mode in ("rrf", "rrf_rerank"):
        with pytest.raises(ValueError, match="window"):
            build_search_body(mode, "q", size=10, window=5)
    build_search_body("bm25", "q", size=10, window=5)  # window is irrelevant to single modes


def test_every_declared_mode_builds():
    for mode in MODES:
        assert build_search_body(mode, "q")["size"] == 10


def test_run_query_returns_ids_in_rank_order_with_both_timings():
    client = FakeSearchClient(hits=("z", "a", "m"), took=42)
    clock = iter([1.0, 1.25]).__next__
    run = run_query(client, "idx", "bm25", Query("q7", "some text"), clock=clock)
    assert run.query_id == "q7" and run.ranked_ids == ["z", "a", "m"]
    assert run.took_ms == 42.0 and run.wall_ms == pytest.approx(250.0)
    call = client.calls[0]
    assert call["index"] == "idx" and call["source"] is False and call["size"] == 10


def test_a_retried_query_reports_only_the_successful_attempt_latency():
    client = FakeSearchClient()
    client.errors = [api_error(502)]
    sleeps = []
    clock = iter([0.0, 10.0, 10.25]).__next__  # failed attempt starts at 0; success 10.0 -> 10.25
    run = run_query(client, "idx", "semantic", Query("q", "t"), sleep=sleeps.append, clock=clock)
    assert run.wall_ms == pytest.approx(250.0)  # the failed attempt and the backoff are excluded
    assert sleeps == [2.0] and len(client.calls) == 2


def test_a_permanent_error_is_not_retried():
    client = FakeSearchClient()
    client.errors = [api_error(400)]
    with pytest.raises(Exception) as excinfo:
        run_query(client, "idx", "bm25", Query("q", "t"), sleep=lambda s: None)
    assert excinfo.value.status_code == 400 and len(client.calls) == 1
