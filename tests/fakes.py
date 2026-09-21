"""Just enough of the Elasticsearch client for offline tests, with scriptable failures."""
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import ApiError

from dataset import Document


def api_error(status: int) -> ApiError:
    meta = ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=NodeConfig("http", "localhost", 9200),
    )
    return ApiError(f"status {status}", meta, body={})


def make_docs(n: int) -> list[Document]:
    return [Document(id=str(i), title=f"Title {i}", text=f"body {i}") for i in range(n)]


class FakeIndices:
    def __init__(self):
        self.created: dict[str, dict] = {}

    def exists(self, index):
        return index in self.created

    def create(self, index, **kwargs):
        self.created[index] = kwargs

    def get_mapping(self, index):
        return {index: {"mappings": self.created[index]["mappings"]}}


class FakeClient:
    def __init__(self):
        self.indices = FakeIndices()
        self.store: dict[str, dict[str, dict]] = {}
        self.bulk_calls: list[list[str]] = []  # doc ids sent in each bulk call
        # doc id -> statuses to fail with, consumed one per attempt
        self.item_failures: dict[str, list[int]] = {}
        # exceptions raised by successive bulk calls, consumed in order
        self.request_errors: list[Exception] = []

    def options(self, **kwargs):
        return self

    def mget(self, index, docs):
        present = self.store.get(index, {})
        return {"docs": [{"_id": d["_id"], "found": d["_id"] in present} for d in docs]}

    def bulk(self, operations):
        if self.request_errors:
            raise self.request_errors.pop(0)
        ids, items = [], []
        for meta, source in zip(operations[0::2], operations[1::2]):
            index, doc_id = meta["index"]["_index"], meta["index"]["_id"]
            ids.append(doc_id)
            scripted = self.item_failures.get(doc_id)
            if scripted:
                status = scripted.pop(0)
                error = {"type": "boom", "reason": "scripted"}
                items.append({"index": {"_id": doc_id, "status": status, "error": error}})
            else:
                self.store.setdefault(index, {})[doc_id] = source
                items.append({"index": {"_id": doc_id, "status": 201}})
        self.bulk_calls.append(ids)
        return {"errors": any(i["index"]["status"] >= 300 for i in items), "items": items}


class FakeInference:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # exceptions are raised, anything else succeeds
        self.calls = 0

    def inference(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return {"sparse_embedding": []}


class FakeInferenceClient:
    def __init__(self, outcomes=()):
        self.inference = FakeInference(outcomes)

    def options(self, **kwargs):
        return self


class FakeClock:
    """A clock that only moves when sleep() is called."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
