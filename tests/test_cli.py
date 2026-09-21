"""The command-line layer: argument validation and the guards that run before any request."""
import pytest

import rerank_offline
import run_benchmark
import run_ingest
from config import positive_int


def rejects(parser, *argv):
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2  # argparse's usage-error exit


def test_the_default_modes_do_not_include_elasticsearchs_reranker():
    args = run_benchmark.build_parser().parse_args(["--tag", "dev-x"])
    assert args.modes == ["bm25", "semantic", "rrf"] and "rrf_rerank" not in args.modes
    explicit = run_benchmark.build_parser().parse_args(["--tag", "dev-x", "--modes", "bm25", "rrf_rerank"])
    assert explicit.modes == ["bm25", "rrf_rerank"]  # still available on request


@pytest.mark.parametrize("flag", ["--depth", "--k", "--window", "--latency-queries", "--rank-constant"])
def test_benchmark_counts_of_zero_are_rejected_not_read_as_unset(flag):
    rejects(run_benchmark.build_parser(), "--tag", "dev-x", flag, "0")


def test_depth_is_unset_by_default_and_accepted_when_positive():
    parser = run_benchmark.build_parser()
    assert parser.parse_args(["--tag", "dev-x"]).depth is None
    assert parser.parse_args(["--tag", "dev-x", "--depth", "20"]).depth == 20


def test_ingest_limit_of_zero_is_rejected_instead_of_ingesting_everything():
    parser = run_ingest.build_parser()
    rejects(parser, "--model", "e5", "--limit", "0")
    rejects(parser, "--model", "e5", "--batch-size", "0")
    args = parser.parse_args(["--model", "e5"])
    assert args.limit is None
    assert parser.parse_args(["--model", "e5", "--limit", "5"]).limit == 5


def test_rerank_counts_must_be_positive():
    parser = rerank_offline.build_parser()
    rejects(parser, "--source", "x", "--tag", "dev-x", "--threads", "0")
    rejects(parser, "--source", "x", "--tag", "dev-x", "--candidates", "0")


def test_positive_int_accepts_only_whole_numbers_of_at_least_one():
    assert positive_int("3") == 3
    for bad in ("0", "-2", "1.5", "abc"):
        with pytest.raises(Exception):
            positive_int(bad)


class Boom:
    def __init__(self, *a, **k):
        raise AssertionError("Elasticsearch was contacted before the guards ran")


@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.setenv("ES_PASSWORD", "x")
    monkeypatch.setattr(run_benchmark, "REPO", tmp_path)
    monkeypatch.setattr(run_benchmark, "Elasticsearch", Boom)
    monkeypatch.setattr(run_benchmark, "code_version", lambda repo: {"commit": "abc", "dirty": False})
    return tmp_path


def test_a_missing_password_stops_the_run_before_any_request(monkeypatch, cli):
    monkeypatch.delenv("ES_PASSWORD")
    with pytest.raises(SystemExit, match="ES_PASSWORD"):
        run_benchmark.main(["--tag", "dev-x"])


def test_a_release_tag_from_a_dirty_tree_is_refused_before_any_request(monkeypatch, cli):
    monkeypatch.setattr(run_benchmark, "code_version", lambda repo: {"commit": "abc", "dirty": True})
    with pytest.raises(SystemExit, match="uncommitted changes"):
        run_benchmark.main(["--tag", "release-9"])


def test_a_scratch_tag_from_a_dirty_tree_gets_past_the_guard(monkeypatch, cli):
    monkeypatch.setattr(run_benchmark, "code_version", lambda repo: {"commit": "abc", "dirty": True})
    with pytest.raises(AssertionError, match="contacted"):  # reached the client, i.e. the guard let it through
        run_benchmark.main(["--tag", "dev-x"])


def test_an_existing_run_is_refused_without_overwrite_and_is_not_destroyed_by_a_failed_start(cli):
    leftover = cli / "artifacts" / "dev-x" / "manifest.json"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        run_benchmark.main(["--tag", "dev-x"])
    with pytest.raises(AssertionError, match="contacted"):  # --overwrite passes the guard, then start fails
        run_benchmark.main(["--tag", "dev-x", "--overwrite"])
    assert leftover.exists()  # the previous run survives a start that never got as far as writing


def test_a_tag_that_could_escape_the_artifacts_directory_is_refused(cli):
    with pytest.raises(SystemExit, match="invalid tag"):
        run_benchmark.main(["--tag", "../../elsewhere", "--overwrite"])
