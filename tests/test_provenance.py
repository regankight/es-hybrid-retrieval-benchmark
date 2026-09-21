import pytest

from artifacts import list_modes, write_rankings
from provenance import require_clean_for_release, reset_run_dir, resolve_run_dir

CLEAN = {"commit": "abc", "dirty": False}
DIRTY = {"commit": "abc", "dirty": True}
UNKNOWN = {"commit": None, "dirty": None}


def test_a_release_tag_needs_a_clean_tree_but_a_scratch_tag_does_not():
    require_clean_for_release("release-1", CLEAN)
    require_clean_for_release("dev-anything", DIRTY)
    require_clean_for_release("dev1", UNKNOWN)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        require_clean_for_release("release-1", DIRTY)


def test_a_release_tag_is_refused_when_the_git_state_cannot_be_determined():
    with pytest.raises(SystemExit, match="cannot determine"):
        require_clean_for_release("release-1", UNKNOWN)


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".hidden", "with space", "..", "-lead", "x;rm"])
def test_run_tags_that_could_leave_the_artifacts_directory_are_rejected(tmp_path, bad):
    with pytest.raises(SystemExit, match="invalid tag"):
        resolve_run_dir(tmp_path, bad, overwrite=True)


@pytest.mark.parametrize("good", ["release-1", "dev2", "repro.1", "rerank_1", "A9"])
def test_ordinary_tags_resolve_inside_the_base_directory(tmp_path, good):
    assert resolve_run_dir(tmp_path, good, overwrite=False) == tmp_path / good


def test_a_non_empty_run_directory_is_refused_unless_overwrite_is_given(tmp_path):
    leftover = tmp_path / "dev-x" / "rankings"
    leftover.mkdir(parents=True)
    (leftover / "bm25.json").write_text("{}")  # a crashed run: files but no manifest
    with pytest.raises(SystemExit, match="not empty"):
        resolve_run_dir(tmp_path, "dev-x", overwrite=False)
    assert resolve_run_dir(tmp_path, "dev-x", overwrite=True) == tmp_path / "dev-x"


def test_an_empty_or_missing_run_directory_is_fine(tmp_path):
    (tmp_path / "dev-empty").mkdir()
    assert resolve_run_dir(tmp_path, "dev-empty", overwrite=False) == tmp_path / "dev-empty"
    assert resolve_run_dir(tmp_path, "dev-new", overwrite=False) == tmp_path / "dev-new"


def test_reset_removes_files_from_an_earlier_run_so_stale_modes_cannot_survive(tmp_path):
    run = tmp_path / "run"
    for mode in ("bm25", "semantic", "rrf"):
        write_rankings(run, mode, {"1": ["a"]})
    reset_run_dir(run)
    write_rankings(run, "bm25", {"1": ["b"]})  # the rerun covers one mode only
    assert list_modes(run) == ["bm25"]


def test_reset_of_a_directory_that_does_not_exist_is_a_no_op(tmp_path):
    reset_run_dir(tmp_path / "never-created")
