import pytest
from elastic_transport import ConnectionTimeout

from config import Config
from dataset import Document
from fakes import FakeClient, FakeClock, make_docs
from indexing import (
    MODELS,
    MappingMismatch,
    document_source,
    ensure_index,
    index_definition,
    index_name,
    ingest,
)

INDEX = "test-index"


def test_ingest_indexes_everything_in_batches():
    client, clock = FakeClient(), FakeClock()
    result = ingest(client, INDEX, make_docs(5), batch_size=2, sleep=clock.sleep)
    assert (result.indexed, result.skipped, result.failed) == (5, 0, {})
    assert client.bulk_calls == [["0", "1"], ["2", "3"], ["4"]]


def test_ids_are_the_dataset_ids_so_reindexing_overwrites():
    client, clock = FakeClient(), FakeClock()
    ingest(client, INDEX, make_docs(3), sleep=clock.sleep)
    assert set(client.store[INDEX]) == {"0", "1", "2"}


def test_rerun_is_a_no_op():
    client, clock = FakeClient(), FakeClock()
    ingest(client, INDEX, make_docs(5), batch_size=2, sleep=clock.sleep)
    calls_before = len(client.bulk_calls)
    result = ingest(client, INDEX, make_docs(5), batch_size=2, sleep=clock.sleep)
    assert (result.indexed, result.skipped) == (0, 5)
    assert len(client.bulk_calls) == calls_before  # nothing re-sent, so no re-inference cost


def test_interrupted_run_resumes_where_it_stopped():
    client, clock = FakeClient(), FakeClock()
    client.store[INDEX] = {"0": {}, "1": {}}  # what a killed run left behind
    result = ingest(client, INDEX, make_docs(5), batch_size=2, sleep=clock.sleep)
    assert (result.indexed, result.skipped) == (3, 2)
    assert client.bulk_calls == [["2", "3"], ["4"]]


def test_transient_item_failure_is_retried_and_only_that_document():
    client, clock = FakeClient(), FakeClock()
    client.item_failures = {"1": [429]}
    result = ingest(client, INDEX, make_docs(3), batch_size=3, sleep=clock.sleep)
    assert result.failed == {} and result.indexed == 3
    assert client.bulk_calls == [["0", "1", "2"], ["1"]]  # the retry carried just the failure
    assert clock.sleeps == [2.0]


def test_whole_request_timeout_is_retried():
    client, clock = FakeClient(), FakeClock()
    client.request_errors = [ConnectionTimeout("timed out")]
    result = ingest(client, INDEX, make_docs(3), batch_size=3, sleep=clock.sleep)
    assert (result.indexed, result.failed) == (3, {})


def test_permanent_failure_is_reported_not_retried_and_others_still_index():
    client, clock = FakeClient(), FakeClock()
    client.item_failures = {"1": [400]}
    result = ingest(client, INDEX, make_docs(3), batch_size=3, sleep=clock.sleep)
    assert list(result.failed) == ["1"] and "400" in result.failed["1"]
    assert result.indexed == 2
    assert len(client.bulk_calls) == 1  # no retry for a 400
    assert clock.sleeps == []


def test_exhausted_retries_are_reported_with_the_last_reason():
    client, clock = FakeClient(), FakeClock()
    client.item_failures = {"1": [429, 429, 429]}
    result = ingest(client, INDEX, make_docs(2), batch_size=2, attempts=3, sleep=clock.sleep)
    assert list(result.failed) == ["1"] and "429" in result.failed["1"]
    assert result.indexed == 1
    assert clock.sleeps == [2.0, 4.0]


def test_unexpected_errors_are_not_swallowed():
    client, clock = FakeClient(), FakeClock()
    client.request_errors = [ValueError("a bug, not a blip")]
    with pytest.raises(ValueError):
        ingest(client, INDEX, make_docs(2), sleep=clock.sleep)


def test_progress_callback_fires_once_per_batch():
    client, clock, seen = FakeClient(), FakeClock(), []
    ingest(
        client, INDEX, make_docs(5), batch_size=2, sleep=clock.sleep,
        on_progress=lambda result, total: seen.append((result.indexed, total)),
    )
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_document_source_joins_title_and_abstract_for_the_semantic_field():
    assert document_source(Document("1", "A title", "the abstract")) == {
        "title": "A title",
        "text": "the abstract",
        "semantic": "A title. the abstract",
    }


def test_document_source_without_a_title_has_no_stray_punctuation():
    assert document_source(Document("1", "", "the abstract"))["semantic"] == "the abstract"


def test_ensure_index_creates_one_shard_no_replicas_with_the_models_endpoint():
    client = FakeClient()
    assert ensure_index(client, "idx-e5", MODELS["e5"]) is True
    created = client.indices.created["idx-e5"]
    assert created["settings"] == {"number_of_shards": 1, "number_of_replicas": 0}
    assert created["mappings"]["properties"]["semantic"]["inference_id"] == MODELS["e5"].inference_id


def test_ensure_index_reuses_a_matching_index():
    client = FakeClient()
    ensure_index(client, "idx-e5", MODELS["e5"])
    assert ensure_index(client, "idx-e5", MODELS["e5"]) is False


def test_ensure_index_refuses_an_index_built_for_another_model():
    client = FakeClient()
    ensure_index(client, "idx", MODELS["elser"])
    with pytest.raises(MappingMismatch):
        ensure_index(client, "idx", MODELS["e5"])


def test_index_name_is_prefix_plus_model_key():
    assert index_name(Config(index_name="bench"), MODELS["elser"]) == "bench-elser"
    assert index_definition(MODELS["e5"])["mappings"]["properties"]["title"] == {"type": "text"}
