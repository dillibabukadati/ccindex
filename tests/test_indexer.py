# tests/test_indexer.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from ccindex.config import Config
from ccindex.indexer import Indexer


def _make_mock_model():
    model = MagicMock()
    model.embed.side_effect = lambda texts: np.random.randn(len(texts), 768).astype(np.float32)
    return model


def test_full_index_creates_db(tmp_path):
    (tmp_path / "app.py").write_text("def foo(): pass\n")
    model = _make_mock_model()

    indexer = Indexer(root=tmp_path, config=Config(), model=model)
    indexer.run_full()

    assert (tmp_path / ".ccindex" / "index.db").exists()


def test_full_index_adds_ccindex_to_gitignore(tmp_path):
    (tmp_path / "app.py").write_text("def foo(): pass\n")
    model = _make_mock_model()

    indexer = Indexer(root=tmp_path, config=Config(), model=model)
    indexer.run_full()

    gitignore = tmp_path / ".gitignore"
    assert gitignore.exists()
    assert ".ccindex/" in gitignore.read_text()


def test_full_index_does_not_duplicate_gitignore_entry(tmp_path):
    (tmp_path / ".gitignore").write_text(".ccindex/\n")
    (tmp_path / "app.py").write_text("def foo(): pass\n")
    model = _make_mock_model()

    indexer = Indexer(root=tmp_path, config=Config(), model=model)
    indexer.run_full()

    text = (tmp_path / ".gitignore").read_text()
    assert text.count(".ccindex/") == 1


def test_incremental_skips_unchanged_files(tmp_path):
    (tmp_path / "app.py").write_text("def foo(): pass\n")
    model = _make_mock_model()

    indexer = Indexer(root=tmp_path, config=Config(), model=model)
    indexer.run_full()
    call_count_after_full = model.embed.call_count

    indexer.run_incremental()
    assert model.embed.call_count == call_count_after_full


def test_incremental_reindexes_changed_file(tmp_path):
    src = tmp_path / "app.py"
    src.write_text("def foo(): pass\n")
    model = _make_mock_model()

    indexer = Indexer(root=tmp_path, config=Config(), model=model)
    indexer.run_full()
    call_count_before = model.embed.call_count

    import time; time.sleep(0.01)
    src.write_text("def foo(): return 1\ndef bar(): pass\n")

    indexer.run_incremental()
    assert model.embed.call_count > call_count_before


def test_incremental_removes_deleted_file(tmp_path):
    src = tmp_path / "app.py"
    src.write_text("def foo(): pass\n")
    model = _make_mock_model()

    indexer = Indexer(root=tmp_path, config=Config(), model=model)
    indexer.run_full()

    src.unlink()
    indexer.run_incremental()

    from ccindex.index import Index
    idx = Index(tmp_path / ".ccindex" / "index.db")
    assert "app.py" not in idx.get_all_mtimes()
