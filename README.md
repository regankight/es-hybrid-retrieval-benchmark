# Hybrid retrieval on Elasticsearch: do fusion and reranking actually pay off?

A measured comparison of four retrieval strategies on a labeled scientific-retrieval dataset
(BEIR SciFact): keyword search (BM25), dense search, hybrid search fused with Reciprocal Rank
Fusion (RRF), and hybrid plus a cross-encoder reranker. Every number below has a confidence
interval, latency is measured on held-out queries so a cache cannot flatter it, and every
result can be regenerated from the artifacts committed in this repository.

This is a reference implementation and benchmark, not a production service. The limits are
stated in full below.

## Bottom line

**Recommendation: ship RRF hybrid (BM25 plus dense), and do not add an off-the-shelf reranker.**

- **Hybrid search is a real gain.** nDCG@10 goes from 0.654 (BM25) to 0.702 (RRF): a paired
  difference of **+0.049, 95% CI [+0.028, +0.069]**. It also beats dense search alone by +0.042
  [+0.015, +0.069]. Cost: about 82 ms on a cold query, 4 ms when the query embedding is cached.
- **Dense search alone is no better than BM25 on this dataset.** +0.007 [-0.030, +0.044], with
  69 queries won and 66 lost. That fits BEIR, where several dense models trail BM25 on SciFact
  (for example TAS-B at 0.643 against BM25's 0.665).
- **A small cross-encoder reranker adds cost without a measurable gain.** Reranking RRF's top 20
  scores 0.694, a difference of -0.008 [-0.035, +0.020] from RRF, at +357 ms per query. It fixes
  many of fusion's failures and breaks others, for no net change (details below).
- **Latency numbers are easy to get wrong.** Elasticsearch caches query embeddings, so the same
  query is about 45x faster the second time in the dense mode. My first version of this benchmark
  reported the hybrid mode as faster than the dense search it contains. This one is built so
  that cannot happen.

## Results

Setup: 5,183 abstracts, 300 test queries with 339 relevance judgments, Elasticsearch 9.5.2 on an
Apple-silicon laptop (Docker; 10 CPUs, about 8 GB), run on 2026-09-21. Intervals are 95%
bootstrap intervals over queries (10,000 resamples, fixed seed). Source:
[`artifacts/release-1/report.md`](artifacts/release-1/report.md).

### Retrieval quality

| Mode | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|
| BM25 | 0.654 [0.607, 0.698] | 0.784 [0.737, 0.828] | 0.620 [0.571, 0.668] |
| Dense (`multilingual-e5-small`) | 0.661 [0.614, 0.705] | 0.775 [0.728, 0.820] | 0.633 [0.584, 0.681] |
| **RRF hybrid** | **0.702** [0.659, 0.745] | **0.833** [0.790, 0.872] | **0.670** [0.623, 0.717] |
| RRF + cross-encoder rerank | 0.694 [0.650, 0.736] | 0.827 [0.784, 0.866] | 0.661 [0.615, 0.707] |

The per-mode intervals overlap heavily, which is exactly why the comparison that matters is
paired: both modes are scored on the same queries, so a hard query is hard for both.

### Is the difference real?

Paired difference in mean nDCG@10 (candidate minus baseline), same 300 queries:

| Candidate vs baseline | Difference [95% CI] | Excludes 0? | Wins | Losses | Ties |
|---|---|---|---|---|---|
| RRF vs BM25 | +0.049 [+0.028, +0.069] | yes | 78 | 32 | 190 |
| RRF vs dense | +0.042 [+0.015, +0.069] | yes | 71 | 40 | 189 |
| Dense vs BM25 | +0.007 [-0.030, +0.044] | no | 69 | 66 | 165 |
| Rerank vs RRF | -0.008 [-0.035, +0.020] | no | 59 | 56 | 185 |
| Rerank vs BM25 | +0.040 [+0.013, +0.069] | yes | 71 | 41 | 188 |
| Rerank vs dense | +0.034 [+0.001, +0.066] | yes | 72 | 54 | 174 |

### Latency

200 held-out queries per mode, each mode on its own disjoint slice (details under
[How it was measured](#how-it-was-measured)). Client-side milliseconds; p50 with a 95% interval.

| Mode | Cold p50 | Cold p95 | Cached p50 | Cache speedup |
|---|---|---|---|---|
| BM25 | 3 [3, 4] | 5 | 3 | 1.3x |
| Dense | 87 [84, 91] | 147 | 2 | 45.5x |
| RRF hybrid | 82 [78, 91] | 140 | 4 | 20.9x |
| Rerank stage, added on top of RRF | 357 [356, 358] | 373 | n/a | n/a |

An identical repeat run ([`artifacts/repro-1`](artifacts/repro-1)), after another restart of
Elasticsearch, produced **byte-identical rankings** for all three modes and cold medians within
1 to 2 ms (dense 88, RRF 84). BM25 needs no embedding, so the cache barely touches it.

### Reranking: who it helps and who it hurts

The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Apache-2.0, 22.7M parameters, run on CPU)
re-scores the top 20 candidates that RRF saved. It is not Elastic's reranker (see
[Findings](#findings-that-would-change-a-decision)); it is a different, much smaller model.

| Queries where | n | RRF nDCG@10 | With rerank | Change |
|---|---|---|---|---|
| fusion did worse than the better single mode | 72 | 0.412 | 0.537 | **+0.125** |
| fusion beat both single modes | 22 | 0.750 | 0.628 | **-0.122** |
| RRF tied the better single mode | 206 | 0.799 | 0.756 | -0.043 |

It repairs the queries where fusion went wrong and damages the ones fusion got right: a
different error profile, not a better one. Headroom is also small. RRF's recall is 0.833 at
the cutoff and 0.880 across the whole 20-candidate pool, 33 queries have no relevant document
anywhere in the pool, and only 12 have one beyond the cutoff that reranking could pull in.

### Approximate-search effort: what `num_candidates` buys

Dense search in Elasticsearch is approximate (HNSW), and `num_candidates` sets how many
candidates it collects before returning the best few. [`ann_sweep.py`](ann_sweep.py) sweeps it
from 20 to 10,000 (Elasticsearch's maximum, used as the reference) for dense-only and for RRF,
whose dense leg is the same search ([`artifacts/ann-1`](artifacts/ann-1), full tables in its
[report](artifacts/ann-1/report.md)). Latency here is vector-search cost only: every query
embedding is already cached.

| `num_candidates` | Dense top-10 overlap with reference | Dense nDCG@10 | Dense client p50 | RRF client p50 |
|---|---|---|---|---|
| 20, 30, 50 (identical results) | 0.978 (57 of 300 queries differ) | 0.6605 | 1.3 to 1.6 ms | 3.5 ms |
| 100 | 0.989 (31 differ) | 0.6613 | 1.4 ms | 3.6 ms |
| 500 | 0.999 (2 differ) | 0.6647 | 1.6 ms | 4.1 ms |
| 1,000 | 1.000 (0 differ) | 0.6647 | 1.7 ms | 4.2 ms |
| 10,000 (reference) | 1.000 | 0.6647 | 1.9 ms | 4.6 ms |

- **The plain `semantic` query used everywhere else in this repo sits in the flat, low end.** Its
  top-10 equals the `num_candidates` 20, 30 and 50 results on all 300 queries, so the main
  benchmark's dense mode misses about 2% of the near-exact top-10 (RRF: 1.6%). I do not know why
  20, 30 and 50 give identical results; I did not investigate.
- **Buying that back is cheap here and changes no conclusion.** Near-exact recall (500 to 1,000)
  costs up to about 0.4 ms for dense (1.3 to 1.6 ms becomes 1.6 to 1.7) and 0.6 to 0.7 ms for RRF (3.5 becomes 4.1 to 4.2). The nDCG@10 difference
  between the default and the reference is -0.0042 for dense (paired 95% interval -0.0121 to
  +0.0005) and -0.0002 for RRF (-0.0066 to +0.0063), so neither is distinguishable from zero.
  nDCG is not monotone in recall either: RRF at 200 scored 0.6994 against 0.7024 at 1,000.
- **What this does not show.** The corpus is 5,183 documents on one shard, so exhaustive-ish search
  is nearly free, and the cost curve will not extrapolate to millions of vectors. The reference is
  the largest setting Elasticsearch allows, not a proven-exact search. I did not test HNSW build
  parameters, quantization, filtered kNN, or concurrent load.

## Findings that would change a decision

1. **Errors are complementary, which is why fusion works.** BM25 beat the dense model on 66
   queries and the dense model beat BM25 on 69. Keyword search wins on rare entity names and
   exact terms (a drug called "Tirasemtiv"); dense wins on paraphrase, misspellings, and
   concept links ("one-child policy" finding a paper about rural China).
2. **Fusion is not free.** RRF did worse than the better single mode on 72 queries (24%) and
   better than both on 22 (7%). RRF rewards agreement over confidence, and with a window of 20
   and rank constant 60 that is arithmetic: a document at rank 1 in one list scores 1/61
   (0.0164), while a document at rank 20 in both scores 2/80 (0.025), so anything present in
   both lists outranks anything present in only one. That this explains the 72 is my hypothesis,
   from reading the worst cases. In the three worst, the dense model ranked the labeled paper
   first while BM25 missed it or ranked it 15th, and documents at middling ranks in both lists
   overtook it. That is a handful of cases, not a tested result.
3. **Query-embedding caching dominates latency.** A cold query costs about 87 ms, almost all of
   it embedding the query; a cached one costs about 2 ms. Any benchmark that lets two modes see
   the same query text will reward whichever runs second.
4. **Off-the-shelf reranking needs a reason.** A small cross-encoder trained on web passages
   did not improve on fused SciFact results, and its errors mirror fusion's. The payoff is more
   likely to come from a fine-tuned reranker, weighted fusion, or query routing. Choosing the
   better of BM25 or dense per query *with hindsight* would score 0.754 against RRF's 0.702.
   That needs the labels, so it is an upper bound, but it shows where the room is.
5. **Elastic's built-in reranker was impractical on a small cluster.** In ad hoc tests during
   development (not committed artifacts), `.rerank-v1` at one allocation took about 2.3 to 2.8 s
   per candidate document, so a 20-candidate window (about 50 s) exceeds Elasticsearch's 30 s
   search timeout. The ML memory budget (about 5.7 GB) was full of idle model deployments, which
   blocked autoscaling; in a couple of earlier runs with more free headroom, warm queries took
   roughly 2 to 7 s. This is a capacity-planning finding, not a verdict on the model.

## How it was measured

| | |
|---|---|
| Dataset | SciFact via BEIR: 5,183 abstracts (about 202 words), 300 test queries, 339 judgments |
| Index | One shard, no replicas. `semantic_text` field backed by Elasticsearch's preconfigured `multilingual-e5-small` (384-dim, cosine), written as "title. abstract" |
| Modes | BM25 (`multi_match` over title and text), dense (`semantic`), RRF (both; window 20, rank constant 60), RRF + reranker |
| Quality | nDCG@10, Recall@10, MRR@10 with trec_eval conventions; a judged query missing from a run is an error, not a zero |
| Statistics | Percentile bootstrap over queries; paired differences with win/loss/tie counts |
| Latency | Separate phase on separate queries (below) |
| Determinism | Rankings re-scored on a sample each run; identical across two full runs |
| Provenance | Every run's manifest records code commit, versions, license type, and node hardware; publishing a run from a dirty working tree is refused, and so is a latency run if Elasticsearch has not restarted since an earlier run used the same held-out queries |

**Why latency has its own phase.** Each mode is timed on its own disjoint slice of real
held-out queries (805 available, none scored, none sharing text with a scored query), so every
mode pays full inference on its cold pass. A second pass over the same slice reports the
cached figure as a separate number. Modes are interleaved query by query in a rotating order so
drift lands on every mode evenly. Elasticsearch is restarted before a latency run, because its
embedding cache outlives a run, and the runner enforces it: it compares the node's start time with
the earlier runs' held-out queries and refuses to start if a cache could still hold them. Quality
artifacts contain rankings only, with no timings.

## Engineering notes: failure modes I hit, and what now guards against them

- **A contaminated cold pass.** A release run reused held-out queries from an earlier scratch
  run. The cache still held them, so the dense mode read 28 ms [2, 61] instead of about 87.
  The interval was the giveaway. Cold passes in which more than 5% of queries run at cached
  speed are now flagged as invalid automatically (the guard was verified against the real
  contaminated data), and a run that would reuse queries the cache still holds is refused up front.
- **Retries that made things worse.** A timed-out rerank request keeps running server-side, so
  retrying queued more work behind a saturated model, and one probe sat stuck for 15 minutes.
  Retry budgets are now small with long backoff, every query records its attempt count, and a
  latency run that needed any retry is flagged invalid, because timing only the successful
  attempt would hide the failures.
- **A test double that encoded an assumption.** The environment collector assumed an API
  returned memory figures it does not, and the fake I wrote for it agreed with the assumption,
  so tests passed while the manifest silently lost the node's hardware. Validating the real
  artifacts caught it. The fake now mirrors responses captured from a live cluster.

## Limitations

- **One dataset, one machine.** SciFact is a small scientific corpus; results may not transfer
  to your data. Latency is single-user on a laptop running Elasticsearch in Docker, with no
  concurrency or throughput testing.
- **Intervals cover query sampling only.** They say nothing about corpus, model, or hardware
  variation. Six comparisons are shown with no multiplicity correction.
- **SciFact labels are sparse.** Most queries have one or two judged documents, so some
  "failures" are relevant documents that were never labeled, and absolute scores are
  conservative. For context, BEIR reports BM25 at 0.665 on SciFact (Table 2, nDCG@10); mine
  is 0.654. I did not investigate the difference (analyzer and query construction are likely).
- **Approximate search is measured at one scale.** The `num_candidates` sweep covers a 5,183-document
  index on one shard, so it says how the trade-off behaves here, not at production size.
- **The dense model is small and untuned.** RRF's window and constant are defaults, not tuned.
  I did not verify whether Elasticsearch adds the `query:` and `passage:` prefixes that E5
  models expect; if it does not, the dense mode is understated.
- **The reranker is off-the-shelf, not fine-tuned, and not Elastic's.** One model, one pool
  size (20), CPU only.
- **ELSER was started and dropped.** It came in as the default path in Elastic's tutorial rather
  than being weighed against the goal, and it indexed at about 0.17 documents per second in the
  real run (roughly 8 hours), so no ELSER results are reported.
- **License.** The ML-backed features ran on a 30-day trial license. I have not verified which
  of them work on a free tier.

## What production would need

TLS and authentication (the local cluster ran without TLS); an ingestion pipeline with chunking
for long documents and incremental updates; multi-node capacity planning for ML models (this
project met the memory-budget and autoscaling limits at small scale); concurrency and
throughput testing; online evaluation or A/B testing; retrieval-quality monitoring; index
versioning and reindexing when the embedding model changes; and a cost model.

**What I would do next:** weighted fusion or query routing (the hindsight gap above is the
size of the prize), tune RRF's window and constant with cross-validation, try a fine-tuned or
domain-adapted reranker, repeat on several BEIR datasets to test generality, and add a
concurrency load test.

## Reproduce

Requires Python 3.12, Docker, and an Elasticsearch 9.5.x with machine learning enabled (I used
[elastic-start-local](https://github.com/elastic/start-local), which starts a trial license).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest                     # 211 offline unit tests; no Elasticsearch needed
cp .env.example .env                           # set ES_PASSWORD, then:  set -a; source .env; set +a

.venv/bin/python run_ingest.py --model e5      # about 95 minutes on my machine; resumable
# restart Elasticsearch here: its embedding cache outlives a run (the runner refuses to start if you forget)
.venv/bin/python run_benchmark.py --tag dev-mine --modes bm25 semantic rrf \
    --latency-queries 200 --stability-sample 100   # tags starting "dev" are scratch and gitignored
```

The committed artifacts let you regenerate the report **without Elasticsearch** (the dataset is
downloaded and checksum-verified on first use):

```bash
.venv/bin/python report.py artifacts/release-1 --rerank artifacts/rerank-1
```

The `num_candidates` sweep needs Elasticsearch; its report regenerates offline:

```bash
.venv/bin/python ann_sweep.py --report artifacts/ann-1 --compare artifacts/release-1
```

The reranking stage runs from saved candidates and needs no Elasticsearch. It lives in its own
environment so the main one stays light (it downloads PyTorch, about 127 MB, and a 91 MB model):

```bash
python3.12 -m venv .venv-rerank && .venv-rerank/bin/pip install -r requirements-rerank.txt
.venv-rerank/bin/python rerank_offline.py --source artifacts/release-1 --tag dev-rerank
```

## Repository layout

| Path | Purpose |
|---|---|
| `dataset.py` | Checksum-verified SciFact download and loader; held-out query set |
| `indexing.py`, `run_ingest.py` | Index definition; resumable, idempotent ingestion |
| `retrieval.py` | Request builders for the four modes, tested without a cluster |
| `metrics.py`, `stats.py` | nDCG, recall, MRR; percentile and paired bootstrap |
| `benchmark.py`, `latency.py`, `run_benchmark.py` | The quality and latency phases and their CLI |
| `rerank_offline.py` | Offline cross-encoder stage over saved candidates |
| `report.py` | Report generator; works from artifacts alone |
| `ann_sweep.py` | `num_candidates` sweep: approximate-search recall, quality and vector-search latency |
| `resilience.py`, `environment.py`, `artifacts.py` | Retry policy, run provenance, on-disk format |
| `provenance.py` | Release guards and run-directory hygiene; standard library only |
| `artifacts/` | `release-1` (the results), `repro-1` (the repeat run), `rerank-1`, `ann-1` (the sweep) |
| `tests/` | 211 offline tests; behaviors that matter were checked by breaking the code and confirming a test fails |
| [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) | The reasoning behind the choices above |

## License and attribution

MIT. Dataset: SciFact (Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP
2020), distributed through BEIR (Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot
Evaluation of Information Retrieval Models", NeurIPS 2021, Datasets and Benchmarks). I checked
the original SciFact repository's license: claim annotations are CC BY 4.0 and abstracts are
ODC-By 1.0, both permitting commercial use with attribution. BEIR's repackaging disclaims any
license guarantee, so the data is downloaded at runtime and never committed. The reranker model
is Apache-2.0.
