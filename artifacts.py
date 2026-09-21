"""On-disk format for benchmark results.

Everything the README reports must be derivable from these files alone. That is the point
of the format: the trial license behind the ML models is temporary, so the results have to
stay checkable after Elasticsearch is gone.

Layout of one run:  artifacts/<tag>/
    manifest.json      what was run, on what, with which code (written by the CLI)
    index.json         the index's mappings and settings as Elasticsearch reported them
    metrics.json       per-mode quality and latency summaries
    runs/<mode>.json   per-query rankings and timings: the source of truth
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retrieval import SearchRun


def write_json(run_dir: Path, name: str, payload: Any) -> None:
    path = Path(run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted keys and a trailing newline keep committed artifacts diff-friendly.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(run_dir: Path, name: str) -> Any:
    return json.loads((Path(run_dir) / name).read_text(encoding="utf-8"))


def write_runs(run_dir: Path, mode: str, runs: list[SearchRun]) -> None:
    payload = {
        "mode": mode,
        "queries": {
            run.query_id: {"ranked": run.ranked_ids, "took_ms": run.took_ms, "wall_ms": run.wall_ms}
            for run in runs
        },
    }
    write_json(run_dir, f"runs/{mode}.json", payload)


def read_runs(run_dir: Path, mode: str) -> dict[str, SearchRun]:
    payload = read_json(run_dir, f"runs/{mode}.json")
    return {
        query_id: SearchRun(query_id, row["ranked"], row["took_ms"], row["wall_ms"])
        for query_id, row in payload["queries"].items()
    }


def list_modes(run_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(run_dir) / "runs").glob("*.json"))
