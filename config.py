"""Configuration from environment variables. Nothing is read from disk implicitly and
no secret has a default: the password is optional here and required only where an
authenticated connection is actually made."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Config:
    es_url: str = "http://localhost:9200"
    es_user: str = "elastic"
    # repr=False keeps the password out of logs, tracebacks, and pytest output.
    es_password: str | None = field(default=None, repr=False)
    index_name: str = "scifact-benchmark"
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            es_url=env.get("ES_URL", defaults.es_url),
            es_user=env.get("ES_USER", defaults.es_user),
            es_password=env.get("ES_PASSWORD") or None,
            index_name=env.get("INDEX_NAME", defaults.index_name),
            data_dir=Path(env.get("DATA_DIR", str(defaults.data_dir))),
        )
