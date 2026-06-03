# tests/test_index.py
import pytest
import numpy as np
from pathlib import Path
from ccindex.index import Index
from ccindex.chunker import Chunk


def _make_chunk(file_path="src/foo.py", symbol="bar", lang="python", mtime=1.0):
    return Chunk(
        file_path=file_path,
        start_line=1, end_line=10,
        symbol=symbol, lang=lang,
        chunk_text=f"def {symbol}(): pass",
        file_mtime=mtime,
    )


def test_schema_created_on_init(tmp_path):
    idx = Index(tmp_path / "index.db")
    tables = idx._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in tables}
    assert "chunks" in names
    assert "meta" in names


def test_wal_mode_enabled(tmp_path):
    idx = Index(tmp_path / "index.db")
    mode = idx._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_upsert_and_retrieve_chunk(tmp_path):
    idx = Index(tmp_path / "index.db")
    chunk = _make_chunk()
    embedding = np.random.randn(768).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    idx.upsert_chunks([(chunk, embedding)])
    mtimes = idx.get_all_mtimes()
    assert "src/foo.py" in mtimes
    assert mtimes["src/foo.py"] == 1.0


def test_delete_by_path(tmp_path):
    idx = Index(tmp_path / "index.db")
    chunk = _make_chunk()
    embedding = np.random.randn(768).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    idx.upsert_chunks([(chunk, embedding)])
    idx.delete_by_path("src/foo.py")
    mtimes = idx.get_all_mtimes()
    assert "src/foo.py" not in mtimes


def test_vector_search_returns_results(tmp_path):
    idx = Index(tmp_path / "index.db")
    chunks = [_make_chunk(symbol=f"fn{i}") for i in range(5)]
    embeddings = [np.random.randn(768).astype(np.float32) for _ in chunks]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    idx.upsert_chunks(list(zip(chunks, embeddings)))

    query_vec = embeddings[0]
    results = idx.vector_search(query_vec, top_k=3)
    assert len(results) <= 3
    assert results[0].symbol == "fn0"


def test_fts_search_returns_results(tmp_path):
    idx = Index(tmp_path / "index.db")
    chunk = _make_chunk(symbol="authenticate_user")
    embedding = np.random.randn(768).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    idx.upsert_chunks([(chunk, embedding)])

    results = idx.fts_search("authenticate", top_k=5)
    assert len(results) >= 1
    assert results[0].symbol == "authenticate_user"


def test_meta_get_set(tmp_path):
    idx = Index(tmp_path / "index.db")
    idx.set_meta("git_commit_hash", "abc123")
    assert idx.get_meta("git_commit_hash") == "abc123"
    assert idx.get_meta("nonexistent") is None


def test_upsert_replaces_existing_chunk(tmp_path):
    idx = Index(tmp_path / "index.db")
    chunk_v1 = _make_chunk(mtime=1.0)
    chunk_v2 = _make_chunk(mtime=2.0)
    emb = np.random.randn(768).astype(np.float32)
    emb /= np.linalg.norm(emb)
    idx.upsert_chunks([(chunk_v1, emb)])
    idx.upsert_chunks([(chunk_v2, emb)])
    mtimes = idx.get_all_mtimes()
    assert mtimes["src/foo.py"] == 2.0
