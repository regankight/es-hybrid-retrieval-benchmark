"""SciFact loader: runtime download from the BEIR mirror, checksum-verified, stdlib only."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"

# query id -> {document id: relevance grade}
Qrels = dict[str, dict[str, int]]

# fetch(url, destination): injectable so tests never touch the network
Fetch = Callable[[str, Path], None]


class ChecksumError(RuntimeError):
    """The downloaded archive does not match the published checksum."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    corpus: list[Document]
    queries: list[Query]
    qrels: Qrels


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _http_fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract, refusing any member that would land outside `target` (zip-slip)."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (target / member).resolve().is_relative_to(target):
                raise ValueError(f"unsafe path in archive: {member!r}")
        zf.extractall(target)


def ensure_scifact(
    data_dir: str | Path,
    *,
    url: str = SCIFACT_URL,
    md5: str = SCIFACT_MD5,
    fetch: Fetch = _http_fetch,
) -> Path:
    """Return the extracted dataset directory, downloading and verifying it if needed.

    Idempotent: an already-extracted dataset is reused as-is. The checksum guards the
    archive at download time; extracted files are not re-hashed on later runs.
    """
    data_dir = Path(data_dir)
    root = data_dir / "scifact"
    required = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    if all((root / name).exists() for name in required):
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "scifact.zip"
    if not (archive.exists() and _md5(archive) == md5):
        # Download to a .part file and only promote it after verification, so a
        # truncated or tampered download can never be mistaken for a valid archive.
        part = archive.with_suffix(".zip.part")
        fetch(url, part)
        actual = _md5(part)
        if actual != md5:
            part.unlink(missing_ok=True)
            raise ChecksumError(f"checksum mismatch for {url}: expected {md5}, got {actual}")
        part.replace(archive)

    _safe_extract(archive, data_dir)
    return root


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(root: Path) -> list[Document]:
    return [
        Document(id=str(row["_id"]), title=row.get("title", ""), text=row.get("text", ""))
        for row in _read_jsonl(root / "corpus.jsonl")
    ]


def load_qrels(root: Path, split: str = "test") -> Qrels:
    """Relevance judgments for a split. Zero-grade rows are dropped: 'judged
    not relevant' must not be counted as relevant."""
    qrels: Qrels = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            grade = int(row["score"])
            if grade > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = grade
    return qrels


def load_queries(root: Path, qrels: Qrels) -> list[Query]:
    """Only queries that have judgments in the chosen split. queries.jsonl also holds
    the other splits' queries, which would be unmeasurable here."""
    return [
        Query(id=str(row["_id"]), text=row["text"])
        for row in _read_jsonl(root / "queries.jsonl")
        if str(row["_id"]) in qrels
    ]


def load_scifact(data_dir: str | Path, split: str = "test", **kwargs) -> Dataset:
    root = ensure_scifact(data_dir, **kwargs)
    qrels = load_qrels(root, split)
    return Dataset(corpus=load_corpus(root), queries=load_queries(root, qrels), qrels=qrels)
