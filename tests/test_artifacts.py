import json

import pytest

from artifacts import (
    list_modes,
    read_json,
    read_latency,
    read_rankings,
    write_json,
    write_latency,
    write_rankings,
)


def test_rankings_round_trip_exactly(tmp_path):
    rankings = {"1": ["a", "b"], "2": []}
    write_rankings(tmp_path, "bm25", rankings)
    assert read_rankings(tmp_path, "bm25") == rankings


def test_a_rankings_file_holds_no_timings_so_flattered_latency_cannot_be_read_from_it(tmp_path):
    write_rankings(tmp_path, "bm25", {"1": ["a"]})
    text = (tmp_path / "rankings" / "bm25.json").read_text()
    assert "ms" not in text and "took" not in text and "wall" not in text


def test_files_are_sorted_and_end_with_a_newline_so_diffs_stay_clean(tmp_path):
    write_json(tmp_path, "x.json", {"b": 1, "a": 2})
    text = (tmp_path / "x.json").read_text()
    assert text.endswith("\n") and text.index('"a"') < text.index('"b"')


def test_write_creates_nested_directories(tmp_path):
    write_json(tmp_path / "deep" / "er", "m.json", {"ok": True})
    assert read_json(tmp_path / "deep" / "er", "m.json") == {"ok": True}


def test_list_modes_reports_what_was_written_in_order(tmp_path):
    for mode in ("rrf", "bm25"):
        write_rankings(tmp_path, mode, {"1": ["a"]})
    assert list_modes(tmp_path) == ["bm25", "rrf"]


def test_a_rankings_file_records_its_own_mode(tmp_path):
    write_rankings(tmp_path, "semantic", {"1": ["a"]})
    assert json.loads((tmp_path / "rankings" / "semantic.json").read_text())["mode"] == "semantic"


def test_latency_round_trips_and_lives_apart_from_rankings(tmp_path):
    payload = {"slice_query_ids": ["h1"], "cold": {"wall_ms": [90.0]}, "warm": {"wall_ms": [2.0]}}
    write_latency(tmp_path, "semantic", payload)
    assert read_latency(tmp_path, "semantic") == {"mode": "semantic", **payload}
    assert (tmp_path / "latency" / "semantic.json").exists()
    assert not (tmp_path / "rankings").exists()


def test_reading_a_mode_that_was_never_written_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_rankings(tmp_path, "bm25")
