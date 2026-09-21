"""What the benchmark ran on. Latency figures are meaningless without this."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Callable

import elasticsearch


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


def collect_environment(client) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "elasticsearch_client": elasticsearch.__versionstr__,
        "elasticsearch_server": _try(lambda: client.info()["version"]["number"]),
        "license": _try(
            lambda: {
                key: client.license.get()["license"].get(key)
                for key in ("type", "status", "expiry_date")
            }
        ),
        "node": _try(lambda: _node(client)),
    }


def code_version(repo_dir: str | Path) -> dict:
    """Commit hash plus whether the tree had uncommitted changes. A dirty tree means the
    numbers cannot be tied back to a commit, which a release run must not allow."""

    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
