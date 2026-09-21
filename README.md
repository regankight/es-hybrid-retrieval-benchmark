# ES Hybrid Retrieval Benchmark — does hybrid search, RRF, and reranking actually beat keyword search?

A small, reproducible benchmark of four Elasticsearch retrieval strategies on a real
labeled dataset: BM25, semantic (`semantic_text`/ELSER), RRF hybrid, and RRF plus a
`text_similarity_reranker`. It reports retrieval quality *and* latency for each, so the
answer to "is reranking worth it?" is a measured tradeoff instead of a demo.

This is a reference implementation and benchmark, not a production service.

**Status: work in progress.** The dataset loader, resumable ingestion, the four retrieval
modes' query builders, and the metrics are done and unit-tested. Ingesting the full corpus
and every benchmark run are still to do. There are no results yet, and none are claimed here.

## Dataset

[SciFact](https://github.com/allenai/scifact) via the [BEIR](https://github.com/beir-cellar/beir)
mirror: 5,183 scientific abstracts (about 202 words on average), 300 test queries, and 339
relevance judgments. The loader downloads and verifies it; the data itself is not committed.

Citations: Wadden et al., "Fact or Fiction: Verifying Scientific Claims" (EMNLP 2020);
Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information
Retrieval Models" (NeurIPS 2021, Datasets and Benchmarks).

## Design decisions

**The dataset is downloaded at runtime and checksum-verified, never committed.** Redistributing
someone else's data invites license questions; downloading it does not. I read the original
SciFact repo's `LICENSE.md`: the claim annotations are CC BY 4.0 and the abstracts are ODC-By 1.0,
both permitting commercial use with attribution (hence the citations above). BEIR's own repackaging
disclaims any license guarantee, so the verification is against the original, and attribution
stays in this README.

**Downloads land in a `.part` file and are promoted only after the MD5 matches.** A truncated or
tampered download can never be mistaken for a valid archive, and a mismatch removes it rather than
leaving it behind to fail confusingly on the next run. Extraction also refuses any archive member
that would land outside the target directory (zip-slip).

**The download function is injectable, so every test runs offline.** The loader takes a `fetch`
callable; tests pass one that copies a tiny fixture archive. I mutation-checked the two behaviors
that matter most (dropping zero-grade judgments, verifying the checksum) by breaking each in a
scratch copy and confirming exactly one test fails.

**A small stdlib loader instead of the `beir` package.** The dataset is three plain files
(`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`). Parsing them takes about thirty lines, and
skipping the dependency avoids pulling in a package for something this small and keeps the
behavior easy to read end to end.

**Queries are restricted to the chosen split, and zero-grade judgments are dropped.**
`queries.jsonl` also holds queries from other splits that have no judgments here and can't be
measured. A judgment of 0 means "judged not relevant" and must not be counted as relevant.

**Configuration comes only from environment variables, with no implicit `.env` loading and no
default for the password.** The password is excluded from `repr`, so it stays out of logs,
tracebacks, and test output. `.env.example` is committed; `.env` is gitignored.

**Python 3.12 is the target.** The optional LangChain integration's published support range tops
out at 3.12 (per its PyPI page at time of writing), so the project targets 3.12 even though
newer interpreters exist.

**One index per semantic model, with one shard and no replicas.** Inference runs at index
time and is slow on a laptop (measured on a 10-core Apple-silicon Mac, Elasticsearch in
Docker: ELSER about 0.45 documents per second, dense `multilingual-e5-small` about 1.2), so
each model gets its own index. Ingesting one never re-runs the other, and either can be
rebuilt alone. One shard on one node keeps BM25's term statistics identical across indices
that hold the same documents.

**The semantic field is written directly as "title. abstract" instead of through
`copy_to`.** It makes exactly what gets embedded explicit and matches the usual
title-plus-abstract convention for this dataset. I did not test the `copy_to` alternative,
so I make no claim about how it would behave.

**Ingestion is resumable and idempotent, and only transient failures are retried.**
Documents keep the dataset's own id, so re-ingesting overwrites instead of duplicating, and
a re-run first skips everything already indexed: an interrupted multi-hour ingestion just
continues. Retries with exponential backoff cover timeouts, 429s, and 502/503/504s. A 400 is
reported immediately instead of being retried. These are not hypothetical: a cold-start 408
and a first-ever-model-import 502 both happened during development.

**"Ready" means the model answered a real request.** With adaptive allocations a model sits
at zero allocations until traffic arrives, so a `started` deployment in the stats API says
nothing about whether the next request will succeed.

**The retrieval modes share one candidate pool.** `window` is how deep each retriever looks
for RRF and how many fused candidates the reranker re-orders, so RRF+rerank re-orders
exactly what plain RRF ranked and the comparison isolates the reranker. Against the live
index, a from-scratch recomputation of RRF (the sum of `1/(60 + rank)` over the BM25 and
semantic lists) reproduced Elasticsearch's fused top 20 in the same order, with a maximum
score difference of 2e-9. That was a sanity check on a partially built index, not a benchmark.

**The reranker reads the `semantic` field (title and abstract), not `text`.** Elasticsearch
accepts both and they rank differently. `semantic` matches what the embedding model sees.
I have not measured which is better.

**Metrics follow trec_eval conventions and evaluation is strict.** Linear gain, a
`1/log2(rank + 1)` discount, and the cutoff applied before scoring keep numbers comparable
with published BEIR results. A judged query that has no run raises an error instead of being
skipped or scored zero, since either would quietly inflate the average. The implementations
are checked against hand-computed values.

**Latency is timed inside the retried call,** so backoff sleeps and failed attempts never
leak into the figures. One caveat learned during development: rerank latency ranged from 1.8
to 19.6 seconds across a handful of runs while ingestion was competing for CPU and the
adaptive scaler was changing allocations (the runs also used different rerank fields, so
this is not a controlled comparison). Latency therefore has to be measured on an otherwise
idle machine after warm-up.

**Latency and quality are measured in separate phases, on different queries.** Elasticsearch
caches query embeddings: in a probe on this setup the same text cost about 90 ms the first
time and about 2 ms the second. An earlier version of this benchmark timed latency while
scoring quality and reported RRF at 5 ms against 83 ms for the semantic search RRF contains,
purely because RRF ran second. So latency now has its own phase. Each mode gets its own
disjoint slice of real held-out queries (never scored, never seen by another mode), so every
mode pays full inference on its cold pass. A second pass over the same slice reports the
cached figure as a separate number. Modes are interleaved query by query with a rotating
order, so slow drift lands on every mode evenly. Quality artifacts hold rankings only, with no
timings, and a test fails if latency ever appears there again. Limits: the slices differ
between modes, so the intervals are unpaired, and a p95 from 100 samples is coarse.

**A query that needed a retry is reported, not absorbed.** Retries exist for cold-start
blips. Against a saturated model they backfire: a timed-out rerank request keeps working
server-side, so each retry queues more work behind it. That happened during development, when
one probe sat retrying for 15 minutes. The retry budget is therefore small with a long backoff,
every query records how many attempts it took, and a latency run that needed any retries is
flagged as invalid, since timing only the successful attempt would hide the failures.

## Running the tests

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## License

MIT — see `LICENSE`.
