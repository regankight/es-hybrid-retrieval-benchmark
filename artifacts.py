"""On-disk format for benchmark results.

Everything the README reports must be derivable from these files alone. That is the point
of the format: the trial license behind the ML models is temporary, so the results have to
stay checkable after Elasticsearch is gone.

Layout of one run:  artifacts/<tag>/
    manifest.json        what was run, on what, with which code (written by the CLI)
    index.json           the index's mappings and settings as Elasticsearch reported them
    metrics.json         per-mode retrieval quality (nDCG, recall, MRR)
    rankings/<mode>.json ranked document ids per scored query: the source of truth for quality
    latency.json         per-mode latency summaries, cold and cached
    latency/<mode>.json  raw per-query timings behind those summaries

Quality and latency live in separate files on purpose: latency measured during a quality
run is distorted by Elasticsearch's query-embedding cache, so it is never recorded there.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

# Mode name shared by the reranking stage that writes it and the report that reads it.
LOCAL_RERANK_MODE = "rrf_rerank_local"


def write_json(run_dir: Path, name: str, payload: Any) -> None:
    path = Path(run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted keys and a trailing newline keep committed artifacts diff-friendly.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(run_dir: Path, name: str) -> Any:
    return json.loads((Path(run_dir) / name).read_text(encoding="utf-8"))


def write_rankings(run_dir: Path, mode: str, rankings: Mapping[str, Sequence[str]]) -> None:
    payload = {
        "mode": mode,
        "queries": {query_id: {"ranked": list(ranked)} for query_id, ranked in rankings.items()},
    }
    write_json(run_dir, f"rankings/{mode}.json", payload)


def read_rankings(run_dir: Path, mode: str) -> dict[str, list[str]]:
    payload = read_json(run_dir, f"rankings/{mode}.json")
    return {query_id: row["ranked"] for query_id, row in payload["queries"].items()}


def list_modes(run_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(run_dir) / "rankings").glob("*.json"))


def write_latency(run_dir: Path, mode: str, payload: Mapping[str, Any]) -> None:
    write_json(run_dir, f"latency/{mode}.json", {"mode": mode, **payload})


def read_latency(run_dir: Path, mode: str) -> dict[str, Any]:
    return read_json(run_dir, f"latency/{mode}.json")
