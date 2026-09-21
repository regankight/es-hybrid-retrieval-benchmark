"""What the benchmark ran on. Latency figures are meaningless without this."""
from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from typing import Callable

import elasticsearch

from provenance import code_version  # noqa: F401  (re-exported; lives in a dependency-free module)


def _try(fn: Callable[[], object]) -> object:
    """Environment facts are context, not the experiment: a failed lookup is recorded
    as unavailable instead of aborting a benchmark run over metadata."""
    try:
        return fn()
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _node(client) -> dict:
    """Hardware as Elasticsearch sees it. Static facts (CPU count, OS) come from nodes.info;
    memory and JVM heap come from nodes.stats, which is where the API actually puts them.
    For a Dockerized node this is the Docker VM's view, not the host Mac's."""
    info = next(iter(client.nodes.info(metric="os,jvm")["nodes"].values()))
    stats = next(iter(client.nodes.stats(metric="os,jvm")["nodes"].values()))
    return {
        "os": info["os"]["pretty_name"],
        "arch": info["os"]["arch"],
        "allocated_processors": info["os"]["allocated_processors"],
        "memory_total_bytes": stats["os"]["mem"]["total_in_bytes"],
        "jvm_version": info["jvm"]["version"],
        "jvm_heap_max_bytes": stats["jvm"]["mem"]["heap_max_in_bytes"],
    }


def _license(client) -> dict:
    license_info = client.license.get()["license"]  # one call: a snapshot, not three
    return {key: license_info.get(key) for key in ("type", "status", "expiry_date")}


def node_started_utc(client) -> datetime:
    """When the Elasticsearch process started. Its query-embedding cache is empty at that moment
    and only a restart empties it, so this is what tells us whether a cache can be trusted cold."""
    node = next(iter(client.nodes.info(metric="jvm")["nodes"].values()))
    return datetime.fromtimestamp(node["jvm"]["start_time_in_millis"] / 1000, tz=timezone.utc)


def collect_environment(client) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "elasticsearch_client": elasticsearch.__versionstr__,
        "elasticsearch_server": _try(lambda: client.info()["version"]["number"]),
        "license": _try(lambda: _license(client)),
        "elasticsearch_started_utc": _try(lambda: node_started_utc(client).isoformat(timespec="seconds")),
        "node": _try(lambda: _node(client)),
    }
