# Code provenance: which commit produced each run

Each run's `manifest.json` records the git commit the code was at when it ran. The history was
rewritten after the runs, so those recorded hashes no longer exist in this repository. This table
maps each one to the commit in today's history that holds the same code.

| Run | Hash in manifest | Code commit today | Tag | Run started (local) | Code commit time |
|---|---|---|---|---|---|
| `release-1` | `fa818df5` | `738c6f3` | `release-1` | 18:24:38 | 18:23:25 |
| `repro-1` | `9aa2cc5` | `258d1db` | `repro-1` | 18:27:01 | 18:26:00 |
| `rerank-1` | `d136d89` | `6695b07` | `rerank-1` | 18:43:50 | 18:43:44 |
| `ann-1` | `030dcba` | `2d04f9d` | `ann-1` | 20:02:42 | 20:02:41 |

All on 2026-09-21, CEST (UTC+2). The manifests record start times in UTC.

**How the mapping was made.** Each run started after its mapped commit and before the next commit.
The gap is between 1 second and 2 minutes. `release-1` records a clean working tree, and `repro-1`
says it ran the same code, with only the release-1 artifacts committed in between. The match is by
timing and commit order, because the original commit objects are gone.

**From now on:** each run is tagged. To check out the code behind a run, use for example
`git checkout release-1`.
