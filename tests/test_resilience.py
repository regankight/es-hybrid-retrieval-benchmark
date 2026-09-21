import pytest
from elastic_transport import ConnectionTimeout

from fakes import FakeClock, FakeInferenceClient, api_error
from resilience import ModelNotReady, is_retryable, retry, wait_for_inference


def test_which_errors_are_retryable():
    assert is_retryable(api_error(408))  # ELSER cold start: model deployment timeout
    assert is_retryable(api_error(502))  # first-ever model import outlasting the proxy
    assert is_retryable(api_error(429))
    assert is_retryable(ConnectionTimeout("timed out"))
    assert not is_retryable(api_error(400))  # a bad request will not fix itself
    assert not is_retryable(api_error(404))
    assert not is_retryable(ValueError("bug"))


def test_retry_backs_off_exponentially_then_returns():
    clock = FakeClock()
    failures = [api_error(503), api_error(503)]

    def flaky():
        if failures:
            raise failures.pop(0)
        return "ok"

    assert retry(flaky, attempts=5, base_delay=2.0, sleep=clock.sleep) == "ok"
    assert clock.sleeps == [2.0, 4.0]


def test_retry_gives_up_after_the_attempt_limit():
    clock = FakeClock()
    calls = []

    def always_fails():
        calls.append(1)
        raise api_error(503)

    with pytest.raises(Exception) as excinfo:
        retry(always_fails, attempts=3, sleep=clock.sleep)
    assert excinfo.value.status_code == 503
    assert len(calls) == 3
    assert len(clock.sleeps) == 2  # no pointless sleep after the final attempt


def test_retry_does_not_retry_permanent_errors():
    clock = FakeClock()
    calls = []

    def bad_request():
        calls.append(1)
        raise api_error(400)

    with pytest.raises(Exception):
        retry(bad_request, attempts=5, sleep=clock.sleep)
    assert len(calls) == 1 and clock.sleeps == []


def test_wait_for_inference_survives_cold_start_timeouts():
    clock = FakeClock()
    client = FakeInferenceClient([api_error(408), api_error(502)])
    waited = wait_for_inference(client, "some-endpoint", poll=5.0, sleep=clock.sleep, clock=clock)
    assert client.inference.calls == 3
    assert waited == 10.0


def test_wait_for_inference_sends_a_query_only_when_given_one():
    # Rerank endpoints reject a request that has no query, embedding endpoints don't need one.
    clock = FakeClock()
    embedding, rerank = FakeInferenceClient(), FakeInferenceClient()
    wait_for_inference(embedding, "e5", sleep=clock.sleep, clock=clock)
    wait_for_inference(rerank, "reranker", query="warm-up query", sleep=clock.sleep, clock=clock)
    assert "query" not in embedding.inference.last_kwargs
    assert rerank.inference.last_kwargs["query"] == "warm-up query"


def test_wait_for_inference_raises_model_not_ready_at_the_deadline():
    clock = FakeClock()
    client = FakeInferenceClient([api_error(408)] * 100)
    with pytest.raises(ModelNotReady, match="some-endpoint"):
        wait_for_inference(client, "some-endpoint", deadline=30.0, poll=10.0, sleep=clock.sleep, clock=clock)


def test_wait_for_inference_surfaces_permanent_errors_immediately():
    clock = FakeClock()
    client = FakeInferenceClient([api_error(404)])  # endpoint does not exist
    with pytest.raises(Exception) as excinfo:
        wait_for_inference(client, "missing", sleep=clock.sleep, clock=clock)
    assert excinfo.value.status_code == 404
    assert client.inference.calls == 1


def test_a_retry_budget_below_one_is_a_clear_error_not_an_assertion():
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        retry(lambda: 1, attempts=0)
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        retry(lambda: 1, attempts=-3)
