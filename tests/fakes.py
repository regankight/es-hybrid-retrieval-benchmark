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
        self.last_kwargs: dict = {}

    def inference(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
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


class FakeSearchClient:
    def __init__(self, hits=("a", "b"), took=7):
        self.calls: list[dict] = []
        self.errors: list[Exception] = []  # raised by successive search calls, in order
        self.hits = hits
        self.took = took

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return {"took": self.took, "hits": {"hits": [{"_id": h} for h in self.hits]}}


def query_text(kwargs: dict) -> str:
    """Dig the query text back out of whatever request body a mode produced."""
    if "query" in kwargs:
        query = kwargs["query"]
        return (query.get("multi_match") or query["semantic"])["query"]
    node = kwargs["retriever"]
    if "text_similarity_reranker" in node:
        node = node["text_similarity_reranker"]["retriever"]
    return node["rrf"]["retrievers"][0]["standard"]["query"]["multi_match"]["query"]


class ScriptedSearchClient:
    """Answers each search by calling hits_for(query_text, request) -> list of doc ids."""

    def __init__(self, hits_for, *, doc_count=0, took=5.0):
        self.hits_for = hits_for
        self.doc_count = doc_count
        self.took = took
        self.calls: list[dict] = []
        self.errors: list[Exception] = []  # raised by successive searches, in order

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        ids = self.hits_for(query_text(kwargs), kwargs)
        return {"took": self.took, "hits": {"hits": [{"_id": i} for i in ids]}}

    def count(self, index):
        return {"count": self.doc_count}


class StepClock:
    """Each successive call returns a value `step` seconds later, so every timed
    attempt (one call at start, one at end) measures exactly `step`."""

    def __init__(self, step=0.01):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


class FakeEnvClient:
    """Answers the handful of metadata calls collect_environment makes."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.license = self
        self.nodes = self

    def _maybe_fail(self, what):
        if what in self.fail:
            raise RuntimeError(f"{what} unavailable")

    # Response shapes below were copied from a live Elasticsearch 9.5.2 node, after an earlier
    # version of this fake (built from an assumption) hid a real KeyError in the collector.
    def info(self, **kwargs):
        if "metric" in kwargs:  # nodes.info(metric=...): static facts, NO memory figures
            self._maybe_fail("nodes")
            return {"nodes": {"n1": {
                "os": {"name": "Linux", "pretty_name": "Red Hat Enterprise Linux 9.8 (Plow)",
                       "arch": "aarch64", "version": "7.0.12-linuxkit",
                       "available_processors": 10, "allocated_processors": 10},
                "jvm": {"version": "26.0.2", "vm_name": "OpenJDK 64-Bit Server VM", "mem": {}},
            }}}
        self._maybe_fail("info")
        return {"version": {"number": "9.5.2"}}

    def stats(self, **kwargs):  # nodes.stats(metric=...): live memory and heap figures
        self._maybe_fail("nodes")
        return {"nodes": {"n1": {
            "os": {"cpu": {"percent": 3}, "mem": {"total_in_bytes": 8_319_504_384}},
            "jvm": {"mem": {"heap_max_in_bytes": 2_147_483_648}},
        }}}

    def get(self):  # license.get()
        self._maybe_fail("license")
        return {"license": {"type": "trial", "status": "active", "expiry_date": "2026-10-01"}}
