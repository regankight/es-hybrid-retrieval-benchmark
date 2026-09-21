"""Run provenance and run-directory hygiene, with no third-party imports.

Kept free of `elasticsearch` on purpose: the optional reranking environment does not install
it, and both stages must apply the same rules about what may be published and where a run
may be written.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def code_version(repo_dir: str | Path) -> dict:
    """Commit hash plus whether the tree had uncommitted changes. A dirty tree means the
    numbers cannot be tied back to a commit, which a release run must not allow."""

    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def require_clean_for_release(tag: str, version: dict) -> None:
    """Tags starting with "dev" are scratch. Any other tag is meant to be committed and
    quoted, so it must come from a clean commit whose state can actually be determined."""
    if tag.startswith("dev"):
        return
    if version["dirty"] is None:
        raise SystemExit("cannot determine the git state of this checkout; use a dev* tag")
    if version["dirty"]:
        raise SystemExit("working tree has uncommitted changes; commit first or use a dev* tag")


def resolve_run_dir(base: str | Path, tag: str, *, overwrite: bool) -> Path:
    """Validate a run tag and return its directory, refusing to reuse a non-empty one.

    The check is "non-empty", not "has a manifest": a run that crashed before writing its
    manifest leaves partial files, and a later run must not silently merge with them.
    """
    if not _TAG.fullmatch(tag):
        raise SystemExit(f"invalid tag {tag!r}: use letters, digits, '.', '_' or '-', starting with a letter or digit")
    run_dir = Path(base) / tag
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{run_dir} is not empty; pick a new tag or pass --overwrite to replace it")
    return run_dir


def reset_run_dir(run_dir: str | Path) -> None:
    """Empty a run directory before reuse, so files from an earlier run (say, modes this run
    did not repeat) cannot survive next to the new manifest. Called only after every
    precondition has passed, so a failed start never destroys the previous run."""
    if Path(run_dir).exists():
        shutil.rmtree(run_dir)
