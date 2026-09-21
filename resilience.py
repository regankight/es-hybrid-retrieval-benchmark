"""Failure handling for a cluster whose ML models scale to zero when idle.

Two real failures motivated this module, both hit while building it: the first request
after idle times out (408) while the model deploys, and a model's very first use imports
it and outlasts even Kibana's proxy (502). Both are transient, so both are retried.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from elastic_transport import ConnectionError as TransportConnectionError
from elastic_transport import ConnectionTimeout
from elasticsearch import ApiError

# 408 request timeout, 429 too many requests, 502/503/504 gateway or availability.
RETRYABLE_STATUSES = frozenset({408, 429, 502, 503, 504})

T = TypeVar("T")


class ModelNotReady(RuntimeError):
    """An inference endpoint did not become usable within the allowed time."""


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ApiError):
        return exc.status_code in RETRYABLE_STATUSES
    # Timeouts and dropped connections only; TLS or serialization errors won't fix themselves.
    return isinstance(exc, (ConnectionTimeout, TransportConnectionError))


def backoff_delay(attempt: int, base_delay: float) -> float:
    """Exponential backoff for a 1-based attempt number: base, 2x, 4x, ..."""
    return base_delay * 2 ** (attempt - 1)


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying only retryable errors, with exponential backoff between tries."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                raise
            sleep(backoff_delay(attempt, base_delay))
    raise AssertionError("unreachable")  # the loop always returns or raises


def wait_for_inference(
    client,
    inference_id: str,
    *,
    query: str | None = None,
    deadline: float = 900.0,
    poll: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Block until the endpoint answers a real inference call; return seconds waited.

    Polling deployment stats is not enough. With adaptive allocations a model sits at
    zero allocations until traffic arrives, so a 'started' deployment says nothing about
    whether the next request will succeed. A tiny real request both wakes the model and
    proves it works. Rerank endpoints also require a `query` to compare the input against.
    """
    start = clock()
    last: Exception | None = None
    request = {"inference_id": inference_id, "input": "warm up", "timeout": "60s"}
    if query is not None:
        request["query"] = query
    while True:
        try:
            client.options(request_timeout=120).inference.inference(**request)
            return clock() - start
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last = exc
        if clock() - start >= deadline:
            raise ModelNotReady(
                f"{inference_id} not ready after {deadline:.0f}s (last error: {last})"
            ) from last
        sleep(poll)
