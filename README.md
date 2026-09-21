# ES Hybrid Retrieval Benchmark — does hybrid search, RRF, and reranking actually beat keyword search?

A small, reproducible benchmark of four Elasticsearch retrieval strategies on a real
labeled dataset: BM25, semantic (`semantic_text`/ELSER), RRF hybrid, and RRF plus a
`text_similarity_reranker`. It reports retrieval quality *and* latency for each, so the
answer to "is reranking worth it?" is a measured tradeoff instead of a demo.

This is a reference implementation and benchmark, not a production service.

**Status: work in progress.** The dataset loader and configuration are done and tested.
There are no benchmark results yet, and none are claimed here.

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

## Running the tests

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## License

MIT — see `LICENSE`.
