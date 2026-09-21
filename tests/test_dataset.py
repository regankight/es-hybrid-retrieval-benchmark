import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest

from dataset import ChecksumError, ensure_scifact, load_scifact

CORPUS = (
    '{"_id": "1", "title": "Alpha", "text": "first abstract"}\n'
    '{"_id": "2", "title": "Beta", "text": "second abstract"}\n'
    '{"_id": "3", "text": "no title on this one"}\n'
)
QUERIES = (
    '{"_id": "10", "text": "query one"}\n'
    '{"_id": "11", "text": "query two"}\n'
    '{"_id": "12", "text": "train-only query"}\n'
)
QRELS_TEST = "query-id\tcorpus-id\tscore\n10\t1\t1\n10\t2\t0\n11\t3\t1\n"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path) -> Path:
    path = tmp_path / "source" / "scifact.zip"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("scifact/corpus.jsonl", CORPUS)
        zf.writestr("scifact/queries.jsonl", QUERIES)
        zf.writestr("scifact/qrels/test.tsv", QRELS_TEST)
    return path


class FakeFetch:
    """Stands in for the network: copies a local file and counts calls."""

    def __init__(self, source: Path):
        self.source = source
        self.calls = 0

    def __call__(self, url: str, dest: Path) -> None:
        self.calls += 1
        shutil.copyfile(self.source, dest)


def test_downloads_verifies_and_extracts(archive, tmp_path):
    fetch = FakeFetch(archive)
    root = ensure_scifact(tmp_path / "data", md5=_md5(archive), fetch=fetch)
    assert fetch.calls == 1
    assert (root / "corpus.jsonl").exists()
    assert (tmp_path / "data" / "scifact.zip").exists()
    assert not (tmp_path / "data" / "scifact.zip.part").exists()


def test_second_call_does_not_fetch_again(archive, tmp_path):
    fetch = FakeFetch(archive)
    ensure_scifact(tmp_path / "data", md5=_md5(archive), fetch=fetch)
    ensure_scifact(tmp_path / "data", md5=_md5(archive), fetch=fetch)
    assert fetch.calls == 1


def test_checksum_mismatch_raises_and_leaves_no_archive(archive, tmp_path):
    fetch = FakeFetch(archive)
    with pytest.raises(ChecksumError):
        ensure_scifact(tmp_path / "data", md5="0" * 32, fetch=fetch)
    data = tmp_path / "data"
    assert not (data / "scifact.zip").exists()
    assert not (data / "scifact.zip.part").exists()
    assert not (data / "scifact").exists()


def test_corrupt_cached_archive_is_redownloaded(archive, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "scifact.zip").write_bytes(b"not a zip")
    fetch = FakeFetch(archive)
    root = ensure_scifact(data, md5=_md5(archive), fetch=fetch)
    assert fetch.calls == 1
    assert (root / "qrels" / "test.tsv").exists()


def test_zip_slip_member_is_rejected(tmp_path):
    evil = tmp_path / "source" / "evil.zip"
    evil.parent.mkdir()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        ensure_scifact(tmp_path / "data", md5=_md5(evil), fetch=FakeFetch(evil))
    assert not (tmp_path / "escaped.txt").exists()


def test_load_filters_queries_to_the_split_and_drops_zero_grades(archive, tmp_path):
    ds = load_scifact(tmp_path / "data", md5=_md5(archive), fetch=FakeFetch(archive))
    assert [q.id for q in ds.queries] == ["10", "11"]  # "12" has no test judgments
    assert ds.qrels == {"10": {"1": 1}, "11": {"3": 1}}  # grade 0 is not relevant


def test_corpus_ids_are_strings_and_missing_title_defaults_to_empty(archive, tmp_path):
    ds = load_scifact(tmp_path / "data", md5=_md5(archive), fetch=FakeFetch(archive))
    assert [d.id for d in ds.corpus] == ["1", "2", "3"]
    assert ds.corpus[2].title == ""
