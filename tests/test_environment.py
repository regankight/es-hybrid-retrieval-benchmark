import subprocess

from datetime import datetime, timezone

from environment import code_version, collect_environment, node_started_utc
from fakes import FakeEnvClient


def test_environment_captures_server_license_and_node_facts():
    env = collect_environment(FakeEnvClient())
    assert env["elasticsearch_server"] == "9.5.2"
    assert env["license"] == {"type": "trial", "status": "active", "expiry_date": "2026-10-01"}
    assert env["node"] == {
        "os": "Red Hat Enterprise Linux 9.8 (Plow)",
        "arch": "aarch64",
        "allocated_processors": 10,
        "memory_total_bytes": 8_319_504_384,  # from nodes.stats: nodes.info has no memory figures
        "jvm_version": "26.0.2",
        "jvm_heap_max_bytes": 2_147_483_648,
    }
    assert env["python"] and env["platform"] and env["cpu_count"]


def test_a_failed_lookup_is_recorded_as_unavailable_and_the_rest_survive():
    env = collect_environment(FakeEnvClient(fail={"license", "nodes"}))
    assert "unavailable" in env["license"] and "unavailable" in env["node"]
    assert env["elasticsearch_server"] == "9.5.2"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def test_code_version_reports_the_commit_and_whether_the_tree_is_dirty(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("one")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    clean = code_version(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False
    (tmp_path / "f.txt").write_text("two")
    assert code_version(tmp_path)["dirty"] is True


def test_code_version_outside_a_repository_is_unknown_not_an_error(tmp_path):
    assert code_version(tmp_path) == {"commit": None, "dirty": None}


def test_one_environment_snapshot_makes_a_single_license_call():
    client, calls = FakeEnvClient(), {"n": 0}
    original = client.get
    def counting():
        calls["n"] += 1
        return original()
    client.get = counting
    collect_environment(client)
    assert calls["n"] == 1  # it used to make three, one per field, so the fields could disagree


def test_the_node_start_time_is_read_as_utc_and_recorded_in_the_environment():
    expected = datetime.fromtimestamp(1790006966.317, tz=timezone.utc)
    assert node_started_utc(FakeEnvClient()) == expected
    assert collect_environment(FakeEnvClient())["elasticsearch_started_utc"] == expected.isoformat(timespec="seconds")
