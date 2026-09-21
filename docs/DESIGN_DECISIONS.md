# Design decisions

The reasoning behind the choices in this project, including the ones I would revisit. The
[README](../README.md) has the results; this file has the "why".

## Data

**The dataset is downloaded at runtime and checksum-verified, never committed.** Redistributing
someone else's data invites license questions; downloading it does not. I read the original
SciFact repository's `LICENSE.md`: claim annotations are CC BY 4.0 and abstracts are ODC-By 1.0,
both permitting commercial use with attribution. BEIR's repackaging disclaims any license
guarantee, so the check is against the original and attribution lives in the README.

**Downloads land in a `.part` file and are promoted only after the MD5 matches.** A truncated
or tampered download can never be mistaken for a valid archive, and a mismatch removes it
rather than leaving it to fail confusingly on the next run. Extraction also refuses any archive
member that would land outside the target directory (zip-slip).

**The download function is injectable, so every test runs offline.** The loader takes a `fetch`
callable; tests pass one that copies a tiny fixture archive.

**A small standard-library loader instead of the `beir` package.** The dataset is three plain
files. Parsing them takes about thirty lines, and skipping the dependency avoids pulling in a
package for something this small and keeps the behavior easy to read end to end.

**Queries are restricted to the chosen split, and zero-grade judgments are dropped.**
`queries.jsonl` also holds queries from other splits that have no judgments here. A judgment of
0 means "judged not relevant" and must not be counted as relevant.

**Held-out queries for latency exclude anything scored.** They exclude every judged query and
every text that also appears among the scored queries, and are de-duplicated by text, so an
embedding cached during a quality run can never leak into a latency measurement.

## Index and ingestion

**One index per semantic model, with one shard and no replicas.** Inference runs at index time
and is slow on a laptop, so each model gets its own index: ingesting one never re-runs the
other, and either can be rebuilt alone. One shard on one node keeps BM25's term statistics
identical across indices that hold the same documents.

**The semantic field is written directly as "title. abstract" instead of through `copy_to`.**
It makes exactly what gets embedded explicit and matches the usual title-plus-abstract
convention for this dataset. I did not test the `copy_to` alternative, so I make no claim about
how it would behave.

**Ingestion is resumable and idempotent, and only transient failures are retried.** Documents
keep the dataset's own id, so re-ingesting overwrites instead of duplicating, and a re-run
first skips everything already indexed: an interrupted multi-hour ingestion just continues.
Retries with exponential backoff cover timeouts, 429s, and 502, 503, and 504. A 400 is reported
immediately instead of being retried. It also refuses to write into an index built for a
different embedding model.

**"Ready" means the model answered a real request.** With adaptive allocations a model sits at
zero allocations until traffic arrives, so a `started` deployment in the stats API says nothing
about whether the next request will succeed.

**ELSER was started and then dropped.** It entered the plan because it was the default path in
Elastic's own tutorial, not because it was weighed against the goal. Measured on this hardware
it indexed at about 0.45 documents per second in isolation and about 0.17 in the real run,
roughly 8 hours for the corpus, because the models compete for a small ML memory budget and the
autoscaler could not add allocations. I stopped it at 560 of 5,183 documents. It is also
Elastic-specific, while dense embeddings are the common case.

**Benchmarking a partial index is refused.** A half-ingested index would score every mode on a
smaller corpus, and the numbers would look plausible while being wrong.

## Retrieval modes

**The modes share one candidate pool.** `window` is how deep each retriever looks for RRF and
how many fused candidates the reranker re-orders, so RRF plus rerank re-orders exactly what plain
RRF ranked and the comparison isolates the reranker. Against the live index, a from-scratch
recomputation of RRF (the sum of `1/(60 + rank)` over the BM25 and dense lists) reproduced
Elasticsearch's fused top 20 in the same order, with a maximum score difference of 2e-9.

**The dense query's depth follows the requested size.** Approximate search can return slightly
different top-10 results when asked for 20 (dense nDCG@10 moved from 0.6598 to 0.6605). The
published run requests 20 so the saved candidates are deep enough for reranking; metrics are
still cut at 10.

## Metrics and statistics

**Metrics follow trec_eval conventions and evaluation is strict.** Linear gain, a
`1/log2(rank + 1)` discount, and the cutoff applied before scoring keep numbers comparable with
published BEIR results. A judged query with no run raises an error instead of being skipped or
scored zero, since either would quietly inflate the average. The implementations are checked
against hand-computed values.

**Paired bootstrap over queries.** Both modes are scored on the same queries, so a hard query is
hard for both; comparing two independent intervals would discard that. Resampling queries
captures query-sampling noise only, which the README states.

## Latency

**Latency and quality are measured in separate phases, on different queries.** Elasticsearch
caches query embeddings: the same text cost about 90 ms the first time and about 2 ms the
second. An earlier version of this benchmark timed latency while scoring quality and reported
RRF at 5 ms against 83 ms for the dense search RRF contains, purely because RRF ran second.
Each mode now gets its own disjoint slice of held-out queries, so every mode pays full
inference on its cold pass; a second pass reports the cached figure separately; and modes are
interleaved query by query with a rotating order so drift lands on every mode evenly. Limits:
the slices differ between modes, so the intervals are unpaired, and a p95 from 200 samples is
coarse.

**The cache outlives a run, so it is detected and cleared.** A restart clears it. Embedding
modes whose cold pass has more than 5% of queries at cached speed are flagged invalid.

**A query that needed a retry is reported, not absorbed.** Timing is taken inside the retried
call so backoff sleeps never leak into the figures, which would also hide failed attempts. So
every query records how many attempts it took and a latency run that needed any retry is
flagged invalid. The retry budget is small with a long backoff, because a timed-out request
against a saturated model keeps working server-side and each retry queues more behind it.

## Reranking

**The reranker runs offline, on saved candidates.** Elasticsearch's own reranker was impractical
on this cluster (see the README), and the trial license behind the cluster is temporary. Stage
one (retrieval) is captured in artifacts and stage two runs from them, so it keeps working
without Elasticsearch, and it is a realistic architecture: reranking is often a separate
service. The trade is that the model is not Elastic's, which the README says wherever it matters.

**The reranker reads "title. abstract", the same text the dense model sees,** and is pinned to an
exact model revision. Ties keep the incoming RRF order, so an uninformative reranker degrades to
plain RRF instead of shuffling it. Its output records a hash of the candidate file it consumed.

## Reproducibility and testing

**Artifacts first.** Every run writes per-query rankings, per-query timings, the index mapping,
and a manifest (code commit, versions, license type, node hardware, dataset checksum). The
report is generated from those files alone, with a fixed bootstrap seed, so the results stay
checkable after the cluster is gone.

**A published run must come from a clean commit.** Tags that do not start with `dev` are refused
when the working tree has uncommitted changes.

**Configuration comes only from environment variables,** with no implicit `.env` loading and no
default for the password. The password is excluded from `repr`, so it stays out of logs,
tracebacks, and test output.

**Behaviors that matter were checked by breaking them.** For the important guarantees (partial
index refused, retries respected, cold-cache detection, paired-difference sign, candidate pool,
and others) I broke the code in a scratch copy and confirmed a test failed. That is a manual
practice, not a mutation-testing framework, and it found real gaps: a missing test that the
latency phase turns the cache check on for embedding modes, and a test double that agreed with
a wrong assumption about an API. Python 3.12 is simply the version this was built and tested on.
