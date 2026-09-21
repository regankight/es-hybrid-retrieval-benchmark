"""Query builders for the four retrieval modes, plus a runner that times them.

The builders are pure: they return request bodies and never touch the network, so the
exact query behind every number in the results is unit-tested rather than assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from dataset import Query
from resilience import retry

MODES = ("bm25", "semantic", "rrf", "rrf_rerank")
DEFAULT_RERANK_INFERENCE_ID = ".rerank-v1-elasticsearch"


def _bm25(text: str) -> dict:
    # Keyword relevance over both fields a reader would search.
    return {"multi_match": {"query": text, "fields": ["title", "text"]}}


def _semantic(text: str) -> dict:
    return {"semantic": {"field": "semantic", "query": text}}


def _filtered(query: dict, filters: Sequence[dict]) -> dict:
    """Wrap in bool/filter only when needed. A filter gates documents without touching
    scores, and is applied inside each retriever so RRF only ever fuses allowed documents."""
    if not filters:
        return query
    return {"bool": {"must": [query], "filter": list(filters)}}


def build_search_body(
    mode: str,
    text: str,
    *,
    size: int = 10,
    window: int = 20,
    rank_constant: int = 60,
    filters: Sequence[dict] = (),
    rerank_inference_id: str = DEFAULT_RERANK_INFERENCE_ID,
    rerank_field: str = "semantic",
) -> dict:
    """Request body for one mode.

    `window` is the candidate pool: how deep each retriever looks for RRF, and how many
    fused candidates the reranker re-orders. RRF and RRF+rerank share it, so the rerank
    mode only re-orders the same pool the plain RRF mode ranked.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if size < 1:
        raise ValueError("size must be at least 1")

    if mode == "bm25":
        return {"size": size, "query": _filtered(_bm25(text), filters)}
    if mode == "semantic":
        return {"size": size, "query": _filtered(_semantic(text), filters)}

    if window < size:
        raise ValueError(f"window ({window}) must be at least size ({size})")
    rrf = {
        "rrf": {
            "retrievers": [
                {"standard": {"query": _filtered(_bm25(text), filters)}},
                {"standard": {"query": _filtered(_semantic(text), filters)}},
            ],
            "rank_window_size": window,
            "rank_constant": rank_constant,
        }
    }
    if mode == "rrf":
        return {"size": size, "retriever": rrf}
    return {
        "size": size,
        "retriever": {
            "text_similarity_reranker": {
                "retriever": rrf,
                "field": rerank_field,
                "inference_id": rerank_inference_id,
                "inference_text": text,
                "rank_window_size": window,
            }
        },
    }


@dataclass(frozen=True)
class SearchRun:
    query_id: str
    ranked_ids: list[str]
    took_ms: float  # server-side time reported by Elasticsearch
    wall_ms: float  # client-side time for the attempt that succeeded


def run_query(
    client,
    index: str,
    mode: str,
    query: Query,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    **options,
) -> SearchRun:
    body = build_search_body(mode, query.text, **options)

    def attempt():
        started = clock()
        response = client.search(index=index, source=False, **body)
        return response, (clock() - started) * 1000

    # Timing is taken inside the retried call so a retry's backoff sleep and the failed
    # attempts never leak into the latency figures.
    response, wall_ms = retry(attempt, sleep=sleep)
    return SearchRun(
        query_id=query.id,
        ranked_ids=[hit["_id"] for hit in response["hits"]["hits"]],
        took_ms=float(response["took"]),
        wall_ms=wall_ms,
    )
