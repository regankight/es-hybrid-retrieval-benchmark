# ANN sweep: ann-1

`num_candidates` swept over [20, 30, 50, 100, 200, 500, 1000, 10000] on `scifact-benchmark-e5` (5183 documents, 300 scored queries). Reference for ANN recall: `num_candidates` = 10000, Elasticsearch's maximum; it is the largest setting available, not a proven-exact search.

Latency is measured with every query embedding already cached, so it is vector-search cost only. 3 timed rounds over all queries. Server-side time is reported by Elasticsearch in whole milliseconds, so it cannot resolve differences below 1 ms; the client column (which includes the network round trip) is the one to read.

Run window 2026-09-21T18:02:42+00:00 to 2026-09-21T18:03:38+00:00; code `030dcba67c`.

## Dense (e5, knn)

| num_candidates | ANN recall@10 | Queries whose top-10 differ from reference | nDCG@10 [95% CI] | Δ nDCG vs reference [95% CI] | Server p50 / p95 (ms) | Client p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| 20 | 0.9777 | 57 | 0.6605 [0.6142, 0.7050] | -0.0042 [-0.0121, +0.0005] | 0.0 / 1.0 | 1.6 / 2.3 |
| 30 | 0.9777 | 57 | 0.6605 [0.6142, 0.7050] | -0.0042 [-0.0121, +0.0005] | 0.0 / 0.0 | 1.4 / 1.9 |
| 50 | 0.9777 | 57 | 0.6605 [0.6142, 0.7050] | -0.0042 [-0.0121, +0.0005] | 0.0 / 0.0 | 1.3 / 1.8 |
| 100 | 0.9890 | 31 | 0.6613 [0.6150, 0.7057] | -0.0033 [-0.0100, +0.0000] | 0.0 / 0.0 | 1.4 / 1.8 |
| 200 | 0.9960 | 12 | 0.6613 [0.6150, 0.7057] | -0.0033 [-0.0100, +0.0000] | 0.0 / 0.0 | 1.4 / 1.9 |
| 500 | 0.9993 | 2 | 0.6647 [0.6184, 0.7091] | +0.0000 [+0.0000, +0.0000] | 0.0 / 1.0 | 1.6 / 2.2 |
| 1000 | 1.0000 | 0 | 0.6647 [0.6184, 0.7091] | +0.0000 [+0.0000, +0.0000] | 0.0 / 1.0 | 1.7 / 2.2 |
| 10000 | 1.0000 | 0 | 0.6647 [0.6184, 0.7091] | +0.0000 [+0.0000, +0.0000] | 1.0 / 1.0 | 1.9 / 2.4 |

Queries (of 300) whose top-10 equal the plain `semantic` query from `artifacts/release-1`, by `num_candidates`: 20: 300, 30: 300, 50: 300, 100: 264, 200: 250, 500: 244, 1000: 243, 10000: 243.

## RRF hybrid

| num_candidates | ANN recall@10 | Queries whose top-10 differ from reference | nDCG@10 [95% CI] | Δ nDCG vs reference [95% CI] | Server p50 / p95 (ms) | Client p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| 20 | 0.9843 | 51 | 0.7022 [0.6586, 0.7451] | -0.0002 [-0.0066, +0.0063] | 2.0 / 4.0 | 3.5 / 5.5 |
| 30 | 0.9843 | 51 | 0.7022 [0.6586, 0.7451] | -0.0002 [-0.0066, +0.0063] | 2.0 / 4.0 | 3.5 / 5.6 |
| 50 | 0.9843 | 51 | 0.7022 [0.6586, 0.7451] | -0.0002 [-0.0066, +0.0063] | 2.0 / 4.0 | 3.5 / 5.7 |
| 100 | 0.9930 | 25 | 0.7018 [0.6580, 0.7447] | -0.0006 [-0.0068, +0.0059] | 2.0 / 4.0 | 3.6 / 5.7 |
| 200 | 0.9977 | 9 | 0.6994 [0.6558, 0.7420] | -0.0030 [-0.0079, +0.0000] | 2.0 / 4.0 | 3.7 / 5.8 |
| 500 | 0.9997 | 1 | 0.7024 [0.6584, 0.7448] | +0.0000 [+0.0000, +0.0000] | 2.0 / 4.0 | 4.1 / 6.3 |
| 1000 | 1.0000 | 0 | 0.7024 [0.6584, 0.7448] | +0.0000 [+0.0000, +0.0000] | 3.0 / 4.0 | 4.2 / 6.3 |
| 10000 | 1.0000 | 0 | 0.7024 [0.6584, 0.7448] | +0.0000 [+0.0000, +0.0000] | 3.0 / 5.0 | 4.6 / 6.6 |

Queries (of 300) whose top-10 equal the main benchmark's RRF from `artifacts/release-1`, by `num_candidates`: 20: 300, 30: 300, 50: 300, 100: 271, 200: 257, 500: 250, 1000: 249, 10000: 249.
