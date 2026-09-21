"""Index definition and resumable, idempotent ingestion.

One index per semantic model. Inference runs at index time and costs hours on a laptop,
so a model gets its own index: ingesting one never re-runs the other, and either can be
rebuilt alone. Every index uses a single shard and no replicas: the cluster is one node,
and one shard keeps BM25 statistics (IDF) identical across indices holding the same
documents, so a keyword score means the same thing everywhere.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from config import Config
from dataset import Document
from resilience import RETRYABLE_STATUSES, backoff_delay, is_retryable, retry


@dataclass(frozen=True)
class SemanticModel:
    key: str
    inference_id: str


MODELS: dict[str, SemanticModel] = {
    "e5": SemanticModel("e5", ".multilingual-e5-small-elasticsearch"),
    "elser": SemanticModel("elser", ".elser-2-elasticsearch"),
}


class MappingMismatch(RuntimeError):
    """An existing index was built for a different semantic model than requested."""


@dataclass
class IngestResult:
    indexed: int = 0
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)  # doc id -> reason


def index_name(cfg: Config, model: SemanticModel) -> str:
    return f"{cfg.index_name}-{model.key}"


def index_definition(model: SemanticModel) -> dict:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
                # Written directly (title + abstract) so exactly what gets embedded is
                # explicit. copy_to from two fields would send two values instead, and
                # how those are embedded was not tested here.
                "semantic": {"type": "semantic_text", "inference_id": model.inference_id},
            }
        },
    }


def ensure_index(client, name: str, model: SemanticModel) -> bool:
    """Create the index if missing. Returns True if created.

    An existing index is reused (that is what makes ingestion resumable) but only if it
    was built for the same model; ingesting into the wrong one would silently mix embeddings.
    """
    if not client.indices.exists(index=name):
        client.indices.create(index=name, **index_definition(model))
        return True
    props = client.indices.get_mapping(index=name)[name]["mappings"]["properties"]
    actual = props.get("semantic", {}).get("inference_id")
    if actual != model.inference_id:
        raise MappingMismatch(
            f"{name} uses inference_id {actual!r}, expected {model.inference_id!r}"
        )
    return False


def document_source(doc: Document) -> dict:
    # Title and abstract joined for the semantic side, the usual retrieval-benchmark convention.
    return {"title": doc.title, "text": doc.text, "semantic": doc.passage()}


def _operations(index: str, docs) -> list[dict]:
    ops: list[dict] = []
    for doc in docs:
        # _id is the dataset's own id: re-indexing a document overwrites, never duplicates.
        ops.append({"index": {"_index": index, "_id": doc.id}})
        ops.append(document_source(doc))
    return ops


def _reason(result: dict) -> str:
    error = result.get("error", {})
    return f"{result['status']} {error.get('type', 'unknown')}: {str(error.get('reason', ''))[:200]}"


def index_batch(
    client,
    index: str,
    docs: Sequence[Document],
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Index a batch, retrying only the documents that failed transiently.

    Returns {doc id: reason} for documents that never made it in. Permanent failures
    (say, a 400) are reported immediately instead of being retried pointlessly.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    pending = {doc.id: doc for doc in docs}
    failed: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for attempt in range(1, attempts + 1):
        try:
            response = client.bulk(operations=_operations(index, pending.values()))
        except Exception as exc:
            if not is_retryable(exc):
                raise
            reasons = {doc_id: f"request failed: {exc}" for doc_id in pending}
        else:
            reasons = {}
            for item in response["items"]:
                result = item["index"]
                doc_id = result["_id"]
                if result["status"] < 300:
                    del pending[doc_id]
                elif result["status"] in RETRYABLE_STATUSES:
                    reasons[doc_id] = _reason(result)
                else:
                    failed[doc_id] = _reason(result)
                    del pending[doc_id]
        if not pending:
            return failed
        if attempt < attempts:
            sleep(backoff_delay(attempt, base_delay))
    failed.update({doc_id: reasons.get(doc_id, "retries exhausted") for doc_id in pending})
    return failed


def _existing_ids(client, index: str, ids: Sequence[str]) -> set[str]:
    response = retry(
        lambda: client.mget(index=index, docs=[{"_id": i, "_source": False} for i in ids])
    )
    return {d["_id"] for d in response["docs"] if d.get("found")}


def ingest(
    client,
    index: str,
    docs: Sequence[Document],
    *,
    batch_size: int = 20,
    attempts: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[IngestResult, int], None] | None = None,
) -> IngestResult:
    """Index `docs`, skipping any already present, so an interrupted run just continues."""
    result = IngestResult()
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        have = _existing_ids(client, index, [d.id for d in batch])
        todo = [d for d in batch if d.id not in have]
        result.skipped += len(batch) - len(todo)
        if todo:
            failed = index_batch(
                client, index, todo, attempts=attempts, base_delay=base_delay, sleep=sleep
            )
            result.failed.update(failed)
            result.indexed += len(todo) - len(failed)
        if on_progress:
            on_progress(result, len(docs))
    return result
