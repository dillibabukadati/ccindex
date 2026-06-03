# tests/test_retrieval.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from ccindex.config import Config
from ccindex.index import Index, SearchResult
from ccindex.retrieval import Retriever, format_hook_output


def _make_result(symbol="foo", file_path="src/foo.py", score=0.8, chunk_id=1):
    return SearchResult(
        chunk_id=chunk_id, file_path=file_path,
        start_line=1, end_line=10,
        symbol=symbol, lang="python",
        chunk_text=f"def {symbol}(): pass",
        score=score,
    )


def _make_retriever(tmp_path, top_k=5, threshold=0.3, token_cap=1500):
    cfg = Config(top_k=top_k, relevance_threshold=threshold, token_cap=token_cap)
    index = Index(tmp_path / "index.db")
    model = MagicMock()
    model.embed.return_value = np.random.randn(1, 768).astype(np.float32)
    reranker = MagicMock()
    reranker.rerank.return_value = [0.9, 0.5, 0.2]
    return Retriever(index=index, model=model, reranker=reranker, config=cfg)


def test_query_returns_empty_when_below_threshold(tmp_path):
    retriever = _make_retriever(tmp_path, threshold=0.5)
    retriever._index.vector_search = MagicMock(return_value=[_make_result(score=0.1)])
    retriever._index.fts_search = MagicMock(return_value=[])
    retriever._reranker.rerank.return_value = [0.1]

    results = retriever.query("some query")
    assert results == []


def test_query_returns_results_above_threshold(tmp_path):
    retriever = _make_retriever(tmp_path, threshold=0.3)
    candidates = [_make_result(symbol=f"fn{i}", score=0.8, chunk_id=i) for i in range(3)]
    retriever._index.vector_search = MagicMock(return_value=candidates)
    retriever._index.fts_search = MagicMock(return_value=[])
    retriever._reranker.rerank.return_value = [0.9, 0.8, 0.7]

    results = retriever.query("some query")
    assert len(results) == 3


def test_format_hook_output_empty_when_no_results():
    assert format_hook_output([]) == ""


def test_format_hook_output_contains_file_and_symbol():
    results = [_make_result(symbol="verify_token", file_path="src/auth.py")]
    output = format_hook_output(results)
    assert "[ccindex context]" in output
    assert "src/auth.py" in output
    assert "verify_token" in output
    assert "[end ccindex context]" in output


def test_token_cap_limits_output(tmp_path):
    retriever = _make_retriever(tmp_path, token_cap=10)
    results = [_make_result(symbol=f"fn{i}") for i in range(10)]
    capped = retriever._apply_token_cap(results)
    total_chars = sum(len(r.chunk_text) for r in capped)
    assert total_chars <= 10 * 4 + 200


def test_dedupe_merges_same_chunk_id(tmp_path):
    retriever = _make_retriever(tmp_path)
    r1 = _make_result(symbol="foo")
    r2 = SearchResult(
        chunk_id=1, file_path="src/foo.py", start_line=1, end_line=10,
        symbol="foo", lang="python", chunk_text="def foo(): pass", score=0.7,
    )
    deduped = retriever._dedupe([r1, r2])
    assert len(deduped) == 1
