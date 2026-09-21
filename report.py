"""Offline report from saved artifacts: intervals, paired comparisons, failure analysis.

    python report.py artifacts/release-1            # writes artifacts/release-1/report.md
    python report.py artifacts/release-1 --rerank artifacts/rerank-1     # adds the reranking section

Needs no Elasticsearch: everything is derived from the run's rankings and latency files,
which is the point of the artifact format. Bootstrap resamples use a fixed seed, so the
report is byte-for-byte reproducible.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from artifacts import LOCAL_RERANK_MODE, list_modes, read_json, read_rankings
from dataset import Document, Query, load_scifact
from metrics import per_query_scores, recall_at_k
from stats import Interval, bootstrap_mean, paired_difference

TIE = 1e-9  # score differences smaller than this count as a tie
MODES = ("bm25", "semantic", "rrf")
RERANK_MODE = LOCAL_RERANK_MODE
LABELS = {"bm25": "BM25", "semantic": "Dense (e5)", "rrf": "RRF hybrid", RERANK_MODE: "RRF + local rerank"}


@dataclass(frozen=True)
class Comparison:
    baseline: str
    candidate: str
    interval: Interval
    wins: int
    losses: int
    ties: int


def compare(baseline: Mapping[str, float], candidate: Mapping[str, float],
            baseline_name: str, candidate_name: str, **kwargs) -> Comparison:
    """Paired interval on the mean difference, plus how many queries improved or got worse."""
    interval = paired_difference(baseline, candidate, **kwargs)
    diffs = [candidate[q] - baseline[q] for q in baseline]
    wins = sum(d > TIE for d in diffs)
    losses = sum(d < -TIE for d in diffs)
    return Comparison(baseline_name, candidate_name, interval, wins, losses, len(diffs) - wins - losses)


def pick_examples(scores: Mapping[str, Mapping[str, float]], per_category: int = 3) -> dict[str, list[str]]:
    """Queries worth reading closely. Deterministic: ordered by margin, ties broken by id."""
    bm25, dense, rrf = scores["bm25"], scores["semantic"], scores["rrf"]
    ids = sorted(bm25)

    def top(margin, keep) -> list[str]:
        chosen = [q for q in ids if keep(margin(q))]
        return sorted(chosen, key=lambda q: (-margin(q), q))[:per_category]

    return {
        "keyword_wins": top(lambda q: bm25[q] - dense[q], lambda m: m > TIE),
        "semantic_wins": top(lambda q: dense[q] - bm25[q], lambda m: m > TIE),
        "fusion_hurts": top(lambda q: max(bm25[q], dense[q]) - rrf[q], lambda m: m > TIE),
        "fusion_helps": top(lambda q: rrf[q] - max(bm25[q], dense[q]), lambda m: m > TIE),
        "all_fail": [q for q in ids if max(bm25[q], dense[q], rrf[q]) <= TIE][:per_category],
    }


def describe(query: Query, judged: Mapping[str, int], rankings: Mapping[str, Mapping[str, list[str]]],
             scores: Mapping[str, Mapping[str, float]], titles: Mapping[str, str]) -> dict:
    """Evidence for one query: where each mode ranked the first relevant document."""
    relevant = [doc for doc, grade in judged.items() if grade > 0]
    per_mode = {}
    for mode, by_query in rankings.items():
        ranked = by_query[query.id]
        first = next((i for i, doc in enumerate(ranked, start=1) if doc in relevant), None)
        per_mode[mode] = {
            "first_relevant_rank": first,
            "ndcg": scores[mode][query.id],
            "top3": [titles.get(doc, doc) for doc in ranked[:3]],
        }
    return {
        "id": query.id,
        "text": query.text,
        "relevant_titles": [titles.get(doc, doc) for doc in relevant],
        "depth": len(next(iter(rankings.values()))[query.id]),
        "modes": per_mode,
    }


def pool_ceiling(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]],
                 pool: int, k: int) -> dict:
    """How much room a reranker over the top-`pool` candidates has: it can only reorder what
    the first stage already retrieved."""
    in_pool = {q: recall_at_k(rankings[q], qrels[q], pool) for q in qrels}
    in_top = {q: recall_at_k(rankings[q], qrels[q], k) for q in qrels}
    return {
        "recall_at_k": sum(in_top.values()) / len(in_top),
        "recall_at_pool": sum(in_pool.values()) / len(in_pool),
        "none_in_pool": sum(v == 0 for v in in_pool.values()),
        "rescuable": sum(1 for q in in_pool if in_pool[q] > 0 and in_top[q] == 0),
    }


def fusion_categories(scores: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    bm25, dense, rrf = scores["bm25"], scores["semantic"], scores["rrf"]
    best = {q: max(bm25[q], dense[q]) for q in bm25}
    return {
        "fusion hurt (RRF below the better single mode)": [q for q in sorted(bm25) if best[q] - rrf[q] > TIE],
        "fusion helped (RRF above both)": [q for q in sorted(bm25) if rrf[q] - best[q] > TIE],
        "RRF tied the better single mode": [q for q in sorted(bm25) if abs(rrf[q] - best[q]) <= TIE],
    }


def breakdown(before: Mapping[str, float], after: Mapping[str, float],
              categories: Mapping[str, Sequence[str]]) -> list[tuple[str, int, float, float]]:
    """Mean score before and after within each category of query."""
    return [(name, len(qs), sum(before[q] for q in qs) / len(qs), sum(after[q] for q in qs) / len(qs))
            for name, qs in categories.items() if qs]


def rerank_examples(before: Mapping[str, float], after: Mapping[str, float], per_category: int = 3) -> dict[str, list[str]]:
    """Queries the reranker improved or damaged most. Deterministic: margin, then id."""
    def top(margin) -> list[str]:
        chosen = [q for q in sorted(before) if margin(q) > TIE]
        return sorted(chosen, key=lambda q: (-margin(q), q))[:per_category]
    return {"rerank_helps": top(lambda q: after[q] - before[q]), "rerank_hurts": top(lambda q: before[q] - after[q])}


def render_rerank_section(info: dict) -> list[str]:
    m, man, k = info["model"], info["manifest"], info["k"]
    out = ["", "## Reranking (offline cross-encoder over saved RRF candidates)", "",
           f"Stage two of a two-stage pipeline: the top {man['candidates']} RRF candidates per query, saved in the source "
           f"run, are re-scored on CPU with `{m['repo']}` (revision `{m['revision'][:10]}`, {m['license']}, "
           f"{m['parameters'] / 1e6:.1f}M parameters, {m['torch_threads']} torch threads). This is NOT Elastic's reranker, "
           f"which was unusable on this cluster. Source candidates verified by hash: `{man['source_rankings_sha256'][:12]}`. "
           f"Code `{str(man['code']['commit'])[:10]}`.", "",
           f"| Mode | nDCG@{k} | Recall@{k} | MRR@{k} |", "|---|---|---|---|"]
    for name, row in (("RRF (same pool)", info["rrf_quality"]), (LABELS[RERANK_MODE], info["quality"])):
        out.append(f"| {name} | {_ci(row[f'ndcg@{k}'])} | {_ci(row[f'recall@{k}'])} | {_ci(row[f'mrr@{k}'])} |")
    out += ["", "| Candidate vs baseline | Difference [95% CI] | Excludes 0? | Wins | Losses | Ties |", "|---|---|---|---|---|---|"]
    for c in info["comparisons"]:
        out.append(f"| {LABELS[c.candidate]} vs {LABELS[c.baseline]} | {c.interval.estimate:+.4f} "
                   f"[{c.interval.low:+.4f}, {c.interval.high:+.4f}] | {'yes' if c.interval.excludes_zero() else 'no'} "
                   f"| {c.wins} | {c.losses} | {c.ties} |")
    lat, ceil = info["latency"], info["ceiling"]
    out += ["", f"**Cost.** Scoring {man['candidates']} candidates takes a median {lat['p50_ms']:.0f} ms per query "
                f"[{lat['p50_ci_low_ms']:.0f}, {lat['p50_ci_high_ms']:.0f}], p95 {lat['p95_ms']:.0f} ms "
                f"(about {lat['p50_ms'] / man['candidates']:.0f} ms per candidate), on top of retrieval. "
                "There is no cache here: each query is scored fresh.", "",
            f"**Headroom.** A reranker can only reorder what the first stage retrieved. RRF's recall is "
            f"{ceil['recall_at_k']:.4f} at the cutoff and {ceil['recall_at_pool']:.4f} across the whole pool. "
            f"{ceil['none_in_pool']} queries have no relevant document anywhere in the pool, and only "
            f"{ceil['rescuable']} have one beyond the cutoff that reranking could pull in.", "",
            f"**Who it helps and who it hurts** (mean nDCG@{k} by kind of query):", "",
            "| Queries where | n | RRF | + rerank | Change |", "|---|---|---|---|---|"]
    for name, n, before, after in info["breakdown"]:
        out.append(f"| {name} | {n} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    out += ["", f"For reference, choosing the better of BM25 and dense per query with hindsight would score "
                f"{info['oracle']:.4f}; that needs the labels, so it is an upper bound, not something achievable.", ""]
    titles = {"rerank_helps": "Where reranking improved most", "rerank_hurts": "Where reranking damaged most"}
    for key, label in titles.items():
        out += [f"### {label}"]
        for ex in info["examples"][key]:
            out += ["", f"**Query {ex['id']}:** {ex['text']}", "", f"Relevant: {'; '.join(ex['relevant_titles'])}", "",
                    f"| Mode | First relevant rank | nDCG@{k} | Top 3 retrieved |", "|---|---|---|---|"]
            for mode in ("bm25", "semantic", "rrf", RERANK_MODE):
                mm = ex["modes"][mode]
                out.append(f"| {LABELS[mode]} | {mm['first_relevant_rank'] or '-'} | {mm['ndcg']:.3f} | "
                           f"{' / '.join(t[:60] for t in mm['top3'])} |")
        if not info["examples"][key]:
            out += ["", "_none_"]
        out.append("")
    return out


def _speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}x"


def _ci(interval: Interval, digits: int = 4) -> str:
    return f"{interval.estimate:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}]"


def render_markdown(manifest: dict, quality: dict, comparisons: Sequence[Comparison], latency: dict | None,
                    categories: dict[str, list[dict]], counts: dict[str, int], n_queries: int, depth: int) -> str:
    env, code = manifest["environment"], manifest["code"]
    k = manifest["settings"]["k"]
    node = env.get("node", {})
    out = [f"# Report: {manifest['tag']}", ""]
    out += [
        f"Generated by `python report.py artifacts/{manifest['tag']}` from saved artifacts only "
        f"(no Elasticsearch needed). Bootstrap: 10,000 resamples, seed 0, 95% intervals.", "",
        "| | |", "|---|---|",
        f"| Code | `{str(code['commit'])[:10]}`, working tree {'dirty' if code['dirty'] else 'clean'} |",
        f"| Elasticsearch | {env['elasticsearch_server']} ({env['license'].get('type')} license), client {env['elasticsearch_client']} |",
        f"| Node | {node.get('allocated_processors')} CPUs, {node.get('memory_total_bytes', 0) / 2**30:.1f} GiB, {node.get('os')} {node.get('arch')} |",
        f"| Corpus / queries | {manifest['dataset']['corpus_documents']} documents, {n_queries} scored queries |",
        f"| Run window | {manifest['started_utc']} to {manifest['finished_utc']} |",
        f"| Note | {manifest['note']} |", "",
    ]

    out += [f"## Retrieval quality ({n_queries} queries, cutoff {k})", "",
            "Per-mode mean with 95% bootstrap interval over queries.", "",
            f"| Mode | nDCG@{k} | Recall@{k} | MRR@{k} |", "|---|---|---|---|"]
    for mode, row in quality.items():
        out.append(f"| {LABELS[mode]} | {_ci(row[f'ndcg@{k}'])} | {_ci(row[f'recall@{k}'])} | {_ci(row[f'mrr@{k}'])} |")

    out += ["", f"## Paired comparisons (nDCG@{k})", "",
            f"Difference in mean nDCG@{k} (candidate minus baseline) on the same queries. "
            "\"Wins\" and \"losses\" count queries whose score rose or fell. Three comparisons are "
            "shown with no multiplicity correction, and the intervals reflect query-sampling noise only.", "",
            "| Candidate vs baseline | Difference [95% CI] | Excludes 0? | Wins | Losses | Ties |", "|---|---|---|---|---|---|"]
    for c in comparisons:
        out.append(f"| {LABELS[c.candidate]} vs {LABELS[c.baseline]} | {c.interval.estimate:+.4f} "
                   f"[{c.interval.low:+.4f}, {c.interval.high:+.4f}] | {'yes' if c.interval.excludes_zero() else 'no'} "
                   f"| {c.wins} | {c.losses} | {c.ties} |")

    if latency is not None:  # a quality-only run has no latency phase
        out += ["", "## Latency", "",
                f"{latency['per_mode']} held-out queries per mode, each mode on its own disjoint slice, "
                f"{latency['order']}. Client-side milliseconds; p50 with 95% bootstrap interval.", "",
                "| Mode | Cold p50 [95% CI] | Cold p95 | Cached p50 | Cache speedup |", "|---|---|---|---|---|"]
        for mode in MODES:
            s = latency["modes"][mode]
            c = s["cold"]
            out.append(f"| {LABELS[mode]} | {c['p50_ms']:.0f} [{c['p50_ci_low_ms']:.0f}, {c['p50_ci_high_ms']:.0f}] "
                       f"| {c['p95_ms']:.0f} | {s['warm']['p50_ms']:.0f} | {_speedup(s['cache_speedup_p50'])} |")

    titles = {
        "keyword_wins": "Where keyword search beat the dense model",
        "semantic_wins": "Where the dense model beat keyword search",
        "fusion_hurts": "Where fusion did worse than the better single mode",
        "fusion_helps": "Where fusion beat both single modes",
        "all_fail": f"Where every mode failed (nDCG@{k} = 0)",
    }
    out += ["", "## Failure analysis", "",
            "Counts over all queries, then the largest cases in each category. "
            "Ranks are the position of the first relevant document; \"-\" means it was not in the saved top "
            f"{depth}.", "",
            "| Category | Queries |", "|---|---|"]
    for key, label in titles.items():
        out.append(f"| {label} | {counts[key]} |")
    for key, label in titles.items():
        out += ["", f"### {label}"]
        if not categories[key]:
            out += ["", "_none_"]
        for ex in categories[key]:
            out += ["", f"**Query {ex['id']}:** {ex['text']}", "",
                    f"Relevant: {'; '.join(ex['relevant_titles'])}", "",
                    f"| Mode | First relevant rank | nDCG@{k} | Top 3 retrieved |", "|---|---|---|---|"]
            for mode in MODES:
                m = ex["modes"][mode]
                out.append(f"| {LABELS[mode]} | {m['first_relevant_rank'] or '-'} | {m['ndcg']:.3f} | "
                           f"{' / '.join(t[:60] for t in m['top3'])} |")
    return "\n".join(out) + "\n"


def build_report(run_dir: Path, corpus: Sequence[Document], queries: Sequence[Query],
                 qrels: Mapping[str, Mapping[str, int]], rerank_dir: Path | None = None) -> str:
    manifest = read_json(run_dir, "manifest.json")
    latency = read_json(run_dir, "latency.json") if (run_dir / "latency.json").exists() else None
    missing = [m for m in MODES if m not in list_modes(run_dir)]
    if missing:
        raise SystemExit(f"{run_dir} has no rankings for {', '.join(missing)}; the report compares bm25, "
                         "semantic and rrf, so run all three")
    k = manifest["settings"]["k"]
    rankings = {mode: read_rankings(run_dir, mode) for mode in MODES}
    per_query = {mode: per_query_scores(rankings[mode], qrels, k) for mode in MODES}

    quality = {
        mode: {metric: bootstrap_mean(list(by_query.values()))
               for metric, by_query in per_query[mode].items()}
        for mode in MODES
    }
    ndcg = {mode: per_query[mode][f"ndcg@{k}"] for mode in MODES}
    comparisons = [compare(ndcg[b], ndcg[c], b, c) for b, c in
                   (("bm25", "rrf"), ("semantic", "rrf"), ("bm25", "semantic"))]

    by_id = {q.id: q for q in queries}
    titles = {doc.id: doc.title for doc in corpus}
    chosen = pick_examples(ndcg)
    categories = {
        key: [describe(by_id[q], qrels[q], rankings, ndcg_by_mode(per_query, k), titles) for q in ids]
        for key, ids in chosen.items()
    }
    counts = {
        "keyword_wins": sum(ndcg["bm25"][q] - ndcg["semantic"][q] > TIE for q in ndcg["bm25"]),
        "semantic_wins": sum(ndcg["semantic"][q] - ndcg["bm25"][q] > TIE for q in ndcg["bm25"]),
        "fusion_hurts": sum(max(ndcg["bm25"][q], ndcg["semantic"][q]) - ndcg["rrf"][q] > TIE for q in ndcg["bm25"]),
        "fusion_helps": sum(ndcg["rrf"][q] - max(ndcg["bm25"][q], ndcg["semantic"][q]) > TIE for q in ndcg["bm25"]),
        "all_fail": sum(max(ndcg[m][q] for m in MODES) <= TIE for q in ndcg["bm25"]),
    }
    depth = len(next(iter(rankings["rrf"].values())))
    text = render_markdown(manifest, quality, comparisons, latency, categories, counts, len(ndcg["bm25"]), depth)
    if rerank_dir is None:
        return text
    text = text.replace(f"`python report.py artifacts/{manifest['tag']}`",
                        f"`python report.py artifacts/{manifest['tag']} --rerank artifacts/{rerank_dir.name}`", 1)

    rr_manifest = read_json(rerank_dir, "manifest.json")
    rankings[RERANK_MODE] = read_rankings(rerank_dir, RERANK_MODE)
    per_query[RERANK_MODE] = per_query_scores(rankings[RERANK_MODE], qrels, k)
    ndcg[RERANK_MODE] = per_query[RERANK_MODE][f"ndcg@{k}"]
    pool = rr_manifest["candidates"]
    pool_scores = per_query_scores({q: rankings["rrf"][q][:pool] for q in qrels}, qrels, k)
    metric_ci = lambda scores: {m: bootstrap_mean(list(v.values())) for m, v in scores.items()}  # noqa: E731
    info = {
        "k": k, "manifest": rr_manifest, "model": rr_manifest["model"],
        "quality": metric_ci(per_query[RERANK_MODE]), "rrf_quality": metric_ci(pool_scores),
        "comparisons": [compare(ndcg[b], ndcg[RERANK_MODE], b, RERANK_MODE) for b in ("rrf", "bm25", "semantic")],
        "latency": read_json(rerank_dir, "latency.json"),
        "ceiling": pool_ceiling(rankings["rrf"], qrels, pool, k),
        "breakdown": breakdown(ndcg["rrf"], ndcg[RERANK_MODE], fusion_categories(ndcg)),
        "oracle": sum(max(ndcg["bm25"][q], ndcg["semantic"][q]) for q in ndcg["bm25"]) / len(ndcg["bm25"]),
        "examples": {
            key: [describe(by_id[q], qrels[q], rankings, ndcg_by_mode(per_query, k), titles) for q in ids]
            for key, ids in rerank_examples(ndcg["rrf"], ndcg[RERANK_MODE]).items()
        },
    }
    return text + "\n".join(render_rerank_section(info)) + "\n"


def ndcg_by_mode(per_query: Mapping[str, Mapping[str, Mapping[str, float]]], k: int) -> dict[str, dict[str, float]]:
    return {mode: per_query[mode][f"ndcg@{k}"] for mode in per_query}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--rerank", type=Path, default=None, help="artifacts dir from rerank_offline.py")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    dataset = load_scifact(args.data_dir)
    report = build_report(args.run_dir, dataset.corpus, dataset.queries, dataset.qrels, args.rerank)
    out = args.out or args.run_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
