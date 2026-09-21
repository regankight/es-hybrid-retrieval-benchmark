"""How much does approximate-search effort (`num_candidates`) buy, and what does it cost?

    python ann_sweep.py --tag ann-1 --compare artifacts/release-1
    python ann_sweep.py --report artifacts/ann-1          # regenerate report.md from the files

Dense search in Elasticsearch is approximate (HNSW). `num_candidates` is how many candidates
each shard collects per query before the best `k` are returned: more candidates find more of
the true nearest neighbours and cost more time. This sweeps it for dense-only and for RRF
hybrid (whose dense leg is the same search) and records, per setting:

  ANN recall   overlap of the top-10 with the top-10 at the largest setting (the reference)
  quality      nDCG@10 / recall@10 against the relevance judgments, paired against the reference
  latency      server-side and client-side time per query

Two design choices matter for reading the numbers. The reference is the largest setting
Elasticsearch allows, not a proven-exact search, so "ANN recall" is relative to it. And the
latency pass runs after every query's embedding is already cached (the quality pass put it
there), so it isolates the vector-search cost, which is the part `num_candidates` changes;
the embedding cost is measured separately in the main benchmark.

The dense query here is an explicit `knn` query with a query-vector builder on the same
semantic_text field, because the plain `semantic` query exposes no `num_candidates`.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from artifacts import read_json, read_rankings, write_json, write_rankings
from dataset import Query
from metrics import per_query_scores, percentile
from resilience import retry
from retrieval import _bm25, _filtered
from stats import bootstrap_mean, paired_difference

MODES = ("dense", "rrf")
LABELS = {"dense": "Dense (e5, knn)", "rrf": "RRF hybrid"}
MAX_NUM_CANDIDATES = 10_000  # Elasticsearch's own ceiling
DEFAULT_CANDIDATES = (20, 30, 50, 100, 200, 500, 1000, MAX_NUM_CANDIDATES)


@dataclass(frozen=True)
class SweepSettings:
    candidates: tuple[int, ...] = DEFAULT_CANDIDATES
    k: int = 10  # metric cutoff
    depth: int = 20  # ranked ids saved; also the knn `k` and the RRF window
    rank_constant: int = 60
    latency_rounds: int = 3  # timed passes over all queries, rotating the setting order
    attempts: int = 3
    retry_delay: float = 5.0

    def __post_init__(self) -> None:
        if len(self.candidates) < 2 or list(self.candidates) != sorted(set(self.candidates)):
            raise ValueError("candidates must be at least two distinct values in ascending order")
        if self.candidates[0] < self.depth:
            raise ValueError(f"num_candidates ({self.candidates[0]}) must be at least depth ({self.depth})")
        if self.candidates[-1] > MAX_NUM_CANDIDATES:
            raise ValueError(f"num_candidates cannot exceed {MAX_NUM_CANDIDATES}")
        if not self.k <= self.depth:
            raise ValueError(f"need k ({self.k}) <= depth ({self.depth})")
        if self.latency_rounds < 1 or self.attempts < 1:
            raise ValueError("latency_rounds and attempts must be at least 1")

    @property
    def reference(self) -> int:
        return self.candidates[-1]


def _knn(text: str, model_id: str, *, k: int, num_candidates: int) -> dict:
    return {"knn": {"field": "semantic", "k": k, "num_candidates": num_candidates,
                    "query_vector_builder": {"text_embedding": {"model_id": model_id, "model_text": text}}}}


def build_ann_body(mode: str, text: str, model_id: str, *, num_candidates: int, depth: int = 20,
                   rank_constant: int = 60, filters: Sequence[dict] = ()) -> dict:
    """Request body for one mode at one `num_candidates`. Pure, like retrieval.build_search_body."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if num_candidates < depth:
        raise ValueError(f"num_candidates ({num_candidates}) must be at least depth ({depth})")
    dense = _filtered(_knn(text, model_id, k=depth, num_candidates=num_candidates), filters)
    if mode == "dense":
        return {"size": depth, "query": dense}
    # The knn *retriever* rejects semantic_text fields; a standard retriever around the knn query works.
    return {"size": depth, "retriever": {"rrf": {
        "retrievers": [{"standard": {"query": _filtered(_bm25(text), filters)}}, {"standard": {"query": dense}}],
        "rank_window_size": depth, "rank_constant": rank_constant}}}


def label(mode: str, num_candidates: int) -> str:
    return f"{mode}_nc{num_candidates}"


def overlap_at_k(ranked: Sequence[str], reference: Sequence[str], k: int) -> float:
    """Share of the reference top-k that this ranking also returns in its top-k."""
    ref = set(reference[:k])
    return len(ref & set(ranked[:k])) / len(ref) if ref else 1.0


@dataclass
class Timings:
    took_ms: list[float]
    wall_ms: list[float]
    retried: int = 0


def _search(client, index: str, body: dict, settings: SweepSettings, *, sleep, clock):
    tries = 0

    def attempt():
        nonlocal tries
        tries += 1
        started = clock()
        response = client.search(index=index, source=False, **body)
        return response, (clock() - started) * 1000

    response, wall = retry(attempt, attempts=settings.attempts, base_delay=settings.retry_delay, sleep=sleep)
    return [h["_id"] for h in response["hits"]["hits"]], float(response["took"]), wall, tries


def run_sweep(client, index: str, queries: Sequence[Query], model_id: str, settings: SweepSettings,
              out_dir: Path, *, sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.perf_counter) -> dict[str, Timings]:
    """Quality pass (also fills the embedding cache), then timed rounds. Writes rankings and raw timings."""
    def body(mode, query, nc):
        return build_ann_body(mode, query.text, model_id, num_candidates=nc, depth=settings.depth,
                              rank_constant=settings.rank_constant)

    for mode in MODES:
        for nc in settings.candidates:
            rankings = {}
            for query in queries:
                ids, _, _, _ = _search(client, index, body(mode, query, nc), settings, sleep=sleep, clock=clock)
                rankings[query.id] = ids
            write_rankings(out_dir, label(mode, nc), rankings)

    timings = {label(m, nc): Timings([], []) for m in MODES for nc in settings.candidates}
    combos = [(m, nc) for m in MODES for nc in settings.candidates]
    for round_index in range(settings.latency_rounds):
        for position, query in enumerate(queries):
            shift = (round_index + position) % len(combos)  # rotate so no setting always runs first
            for mode, nc in combos[shift:] + combos[:shift]:
                _, took, wall, tries = _search(client, index, body(mode, query, nc), settings, sleep=sleep, clock=clock)
                t = timings[label(mode, nc)]
                t.took_ms.append(took)
                t.wall_ms.append(wall)
                t.retried += tries > 1
    write_json(out_dir, "latency.json", {
        "rounds": settings.latency_rounds, "queries": len(queries),
        "settings": {name: {"took_ms": t.took_ms, "wall_ms": t.wall_ms, "retried_queries": t.retried}
                     for name, t in timings.items()}})
    return timings


def summarize(run_dir: Path, qrels: Mapping[str, Mapping[str, int]], settings: Mapping, *,
              compare_dir: Path | None = None) -> dict:
    """Everything the report shows, derived from the saved files alone."""
    k, candidates = settings["k"], settings["candidates"]
    latency = read_json(run_dir, "latency.json")["settings"]
    out: dict = {"modes": {}}
    for mode in MODES:
        ref_rank = read_rankings(run_dir, label(mode, candidates[-1]))
        ref_scores = per_query_scores(ref_rank, qrels, k)["ndcg@" + str(k)]
        known = None
        if compare_dir is not None:
            source = "semantic" if mode == "dense" else "rrf"
            known = read_rankings(compare_dir, source)
        rows = []
        for nc in candidates:
            name = label(mode, nc)
            ranked = read_rankings(run_dir, name)
            scores = per_query_scores(ranked, qrels, k)
            overlap = [overlap_at_k(ranked[q], ref_rank[q], k) for q in ranked]
            took, wall = latency[name]["took_ms"], latency[name]["wall_ms"]
            row = {
                "num_candidates": nc,
                "ann_recall": sum(overlap) / len(overlap),
                "queries_not_identical_to_reference": sum(ranked[q][:k] != ref_rank[q][:k] for q in ranked),
                "ndcg": asdict_interval(bootstrap_mean(list(scores["ndcg@" + str(k)].values()))),
                "recall": sum(scores["recall@" + str(k)].values()) / len(scores["recall@" + str(k)]),
                "ndcg_vs_reference": asdict_interval(paired_difference(ref_scores, scores["ndcg@" + str(k)])),
                "took_p50_ms": percentile(took, 50), "took_p95_ms": percentile(took, 95),
                "wall_p50_ms": percentile(wall, 50), "wall_p95_ms": percentile(wall, 95),
                "retried_queries": latency[name]["retried_queries"],
            }
            if known is not None:
                row["queries_matching_main_benchmark"] = sum(ranked[q][:k] == known[q][:k] for q in ranked)
            rows.append(row)
        out["modes"][mode] = rows
    return out


def asdict_interval(interval) -> dict:
    return {"estimate": interval.estimate, "low": interval.low, "high": interval.high}


def render_markdown(manifest: dict, summary: dict) -> str:
    s = manifest["settings"]
    k, ref = s["k"], s["candidates"][-1]
    out = [f"# ANN sweep: {manifest['tag']}", "",
           f"`num_candidates` swept over {s['candidates']} on `{manifest['index']}` "
           f"({manifest['index_documents']} documents, {manifest['queries']} scored queries). "
           f"Reference for ANN recall: `num_candidates` = {ref}, Elasticsearch's maximum; it is the "
           "largest setting available, not a proven-exact search.", "",
           "Latency is measured with every query embedding already cached, so it is vector-search cost only. "
           f"{s['latency_rounds']} timed rounds over all queries; client-side milliseconds include the "
           "network round trip.", "",
           f"Run window {manifest['started_utc']} to {manifest['finished_utc']}; code `{manifest['code']['commit'][:10]}`.", ""]
    for mode in MODES:
        out += [f"## {LABELS[mode]}", "",
                f"| num_candidates | ANN recall@{k} | Queries whose top-{k} differ from reference | nDCG@{k} [95% CI] "
                f"| Δ nDCG vs reference [95% CI] | Server p50 / p95 (ms) | Client p50 / p95 (ms) |",
                "|---|---|---|---|---|---|---|"]
        for r in summary["modes"][mode]:
            n, d = r["ndcg"], r["ndcg_vs_reference"]
            out.append(f"| {r['num_candidates']} | {r['ann_recall']:.4f} | {r['queries_not_identical_to_reference']} "
                       f"| {n['estimate']:.4f} [{n['low']:.4f}, {n['high']:.4f}] "
                       f"| {d['estimate']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] "
                       f"| {r['took_p50_ms']:.1f} / {r['took_p95_ms']:.1f} | {r['wall_p50_ms']:.1f} / {r['wall_p95_ms']:.1f} |")
        if "queries_matching_main_benchmark" in summary["modes"][mode][0]:
            match = ", ".join(f"{r['num_candidates']}: {r['queries_matching_main_benchmark']}" for r in summary["modes"][mode])
            source = "the plain `semantic` query" if mode == "dense" else "the main benchmark's RRF"
            out += ["", f"Queries (of {manifest['queries']}) whose top-{k} equal {source} from `{manifest['compare']}`, "
                        f"by `num_candidates`: {match}."]
        out.append("")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    from config import positive_int
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", help="run name; becomes artifacts/<tag>/")
    p.add_argument("--report", metavar="RUN_DIR", help="regenerate report.md for an existing run, offline")
    p.add_argument("--candidates", nargs="+", type=positive_int, default=list(DEFAULT_CANDIDATES))
    p.add_argument("--rounds", type=positive_int, default=3, help="timed passes over all queries")
    p.add_argument("--compare", metavar="RUN_DIR", help="a main-benchmark run whose semantic/rrf rankings to match against")
    p.add_argument("--note", default="")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    from elasticsearch import Elasticsearch

    from config import Config
    from dataset import SCIFACT_MD5, load_scifact
    from environment import collect_environment
    from indexing import MODELS, index_name
    from benchmark import verify_index_complete
    from provenance import code_version, require_clean_for_release, reset_run_dir, resolve_run_dir
    from resilience import wait_for_inference

    repo = Path(__file__).resolve().parent
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.tag) == bool(args.report):
        parser.error("give exactly one of --tag (run) or --report (regenerate)")
    try:
        settings = SweepSettings(candidates=tuple(args.candidates), latency_rounds=args.rounds)
    except ValueError as err:
        parser.error(str(err))
    cfg = Config.from_env()
    dataset = load_scifact(cfg.data_dir)
    compare = Path(args.compare) if args.compare else None

    if args.report:
        run_dir = Path(args.report)
        manifest = read_json(run_dir, "manifest.json")
        summary = summarize(run_dir, dataset.qrels, manifest["settings"], compare_dir=compare)
        (run_dir / "report.md").write_text(render_markdown(manifest, summary), encoding="utf-8")
        print(f"wrote {run_dir / 'report.md'}")
        return 0

    if not cfg.es_password:
        raise SystemExit("ES_PASSWORD is not set (see .env.example)")
    version = code_version(repo)
    require_clean_for_release(args.tag, version)
    out_dir = resolve_run_dir(repo / "artifacts", args.tag, overwrite=args.overwrite)
    client = Elasticsearch(cfg.es_url, basic_auth=(cfg.es_user, cfg.es_password), request_timeout=300)
    model = MODELS["e5"]
    index = index_name(cfg, model)
    docs = verify_index_complete(client, index, len(dataset.corpus))
    print(f"waking {model.inference_id} ...", flush=True)
    wait_for_inference(client, model.inference_id)

    reset_run_dir(out_dir)
    started = datetime.now(timezone.utc)
    print(f"sweeping num_candidates {list(settings.candidates)} on {index}: {len(dataset.queries)} queries x "
          f"{len(MODES)} modes, then {settings.latency_rounds} timed rounds ...", flush=True)
    run_sweep(client, index, dataset.queries, model.inference_id, settings, out_dir)
    manifest = {
        "tag": args.tag, "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": args.note,
        "index": index, "index_documents": docs, "queries": len(dataset.queries), "modes": list(MODES),
        "settings": asdict(settings), "compare": args.compare, "model": model.inference_id,
        "dataset": {"name": "BEIR/SciFact", "split": "test", "archive_md5": SCIFACT_MD5},
        "code": version, "environment": collect_environment(client),
    }
    write_json(out_dir, "manifest.json", manifest)
    summary = summarize(out_dir, dataset.qrels, manifest["settings"], compare_dir=compare)
    write_json(out_dir, "summary.json", summary)
    (out_dir / "report.md").write_text(render_markdown(manifest, summary), encoding="utf-8")
    for mode in MODES:
        print(f"\n{LABELS[mode]}\n  {'nc':>6} {'ANN recall':>11} {'nDCG@10':>8} {'server p50':>11} {'p95':>6}")
        for r in summary["modes"][mode]:
            print(f"  {r['num_candidates']:>6} {r['ann_recall']:>11.4f} {r['ndcg']['estimate']:>8.4f} "
                  f"{r['took_p50_ms']:>11.1f} {r['took_p95_ms']:>6.1f}")
        retried = {r["num_candidates"]: r["retried_queries"] for r in summary["modes"][mode] if r["retried_queries"]}
        if retried:
            print(f"  WARNING: latency queries needed retries {retried}; those timings are not clean.")
    print(f"\nartifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
