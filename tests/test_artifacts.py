import json

import pytest

from artifacts import list_modes, read_json, read_runs, write_json, write_runs
from retrieval import SearchRun


def test_runs_round_trip_exactly(tmp_path):
    runs = [SearchRun("1", ["a", "b"], 12.0, 15.5), SearchRun("2", [], 3.0, 4.0)]
    write_runs(tmp_path, "bm25", runs)
    assert read_runs(tmp_path, "bm25") == {r.query_id: r for r in runs}


def test_files_are_sorted_and_end_with_a_newline_so_diffs_stay_clean(tmp_path):
    write_json(tmp_path, "x.json", {"b": 1, "a": 2})
    text = (tmp_path / "x.json").read_text()
    assert text.endswith("\n") and text.index('"a"') < text.index('"b"')


def test_write_creates_nested_directories(tmp_path):
    write_json(tmp_path / "deep" / "er", "m.json", {"ok": True})
    assert read_json(tmp_path / "deep" / "er", "m.json") == {"ok": True}


def test_list_modes_reports_what_was_written_in_order(tmp_path):
    for mode in ("rrf", "bm25"):
        write_runs(tmp_path, mode, [SearchRun("1", ["a"], 1.0, 1.0)])
    assert list_modes(tmp_path) == ["bm25", "rrf"]


def test_a_run_file_records_its_own_mode(tmp_path):
    write_runs(tmp_path, "semantic", [SearchRun("1", ["a"], 1.0, 1.0)])
    assert json.loads((tmp_path / "runs" / "semantic.json").read_text())["mode"] == "semantic"


def test_reading_a_mode_that_was_never_written_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_runs(tmp_path, "bm25")
