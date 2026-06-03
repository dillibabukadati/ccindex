import pytest
from pathlib import Path
from ccindex.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.top_k == 5
    assert cfg.relevance_threshold == 0.3
    assert cfg.token_cap == 1500
    assert cfg.max_file_size_kb == 1024
    assert cfg.batch_size == 32
    assert cfg.max_stale_files == 200
    assert cfg.ignore_patterns == []


def test_project_config_overrides_defaults(tmp_path):
    config_file = tmp_path / ".ccindex" / "config.toml"
    config_file.parent.mkdir()
    config_file.write_text('[query]\ntop_k = 10\nrelevance_threshold = 0.5\n')
    cfg = load_config(project_root=tmp_path)
    assert cfg.top_k == 10
    assert cfg.relevance_threshold == 0.5
    assert cfg.token_cap == 1500  # unchanged default


def test_missing_config_returns_defaults(tmp_path):
    cfg = load_config(project_root=tmp_path)
    assert cfg.top_k == 5


def test_ignore_patterns_merged(tmp_path):
    config_file = tmp_path / ".ccindex" / "config.toml"
    config_file.parent.mkdir()
    config_file.write_text('[ignore]\npatterns = ["migrations/", "*.generated.ts"]\n')
    cfg = load_config(project_root=tmp_path)
    assert "migrations/" in cfg.ignore_patterns
