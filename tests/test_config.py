from pathlib import Path

from config import Config


def test_defaults_when_env_is_empty():
    cfg = Config.from_env({})
    assert cfg.es_url == "http://localhost:9200"
    assert cfg.es_user == "elastic"
    assert cfg.es_password is None
    assert cfg.data_dir == Path("data")


def test_env_overrides_every_field():
    cfg = Config.from_env(
        {
            "ES_URL": "http://es.example:9200",
            "ES_USER": "someone",
            "ES_PASSWORD": "s3cret",
            "INDEX_NAME": "other-index",
            "DATA_DIR": "/tmp/bench-data",
        }
    )
    assert (cfg.es_url, cfg.es_user, cfg.es_password) == ("http://es.example:9200", "someone", "s3cret")
    assert cfg.index_name == "other-index"
    assert cfg.data_dir == Path("/tmp/bench-data")


def test_empty_password_is_treated_as_unset():
    # .env.example ships with `ES_PASSWORD=` — that must not become a real empty password.
    assert Config.from_env({"ES_PASSWORD": ""}).es_password is None


def test_password_never_appears_in_repr():
    cfg = Config.from_env({"ES_PASSWORD": "s3cret"})
    assert "s3cret" not in repr(cfg)


def test_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("INDEX_NAME", "from-os-env")
    assert Config.from_env().index_name == "from-os-env"
