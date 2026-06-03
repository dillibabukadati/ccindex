# tests/test_models.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from ccindex.models import get_model_dir, ModelNotFoundError, EmbeddingModel, Reranker


def test_get_model_dir_finds_package_bundled(tmp_path):
    bundled = tmp_path / "models" / "jina-code-onnx"
    bundled.mkdir(parents=True)
    with patch("ccindex.models._PACKAGE_ROOT", tmp_path):
        result = get_model_dir("jina-code-onnx")
    assert result == bundled


def test_get_model_dir_finds_user_cache(tmp_path):
    user_cache = tmp_path / ".ccindex" / "models" / "jina-code-onnx"
    user_cache.mkdir(parents=True)
    with patch("ccindex.models._PACKAGE_ROOT", tmp_path / "nonexistent"):
        with patch("ccindex.models._USER_CACHE", tmp_path / ".ccindex" / "models"):
            result = get_model_dir("jina-code-onnx")
    assert result == user_cache


def test_get_model_dir_raises_when_missing(tmp_path):
    with patch("ccindex.models._PACKAGE_ROOT", tmp_path / "nonexistent"):
        with patch("ccindex.models._USER_CACHE", tmp_path / "also_nonexistent"):
            with pytest.raises(ModelNotFoundError):
                get_model_dir("jina-code-onnx")


def test_embedding_model_returns_normalized_vectors(tmp_path):
    """Uses a tiny ONNX model mock to verify shape and normalization."""
    mock_session = MagicMock()
    mock_session.run.return_value = [np.random.randn(2, 16, 768).astype(np.float32)]

    mock_tokenizer = MagicMock()
    enc1, enc2 = MagicMock(), MagicMock()
    enc1.ids = [1, 2, 3] + [0] * 13
    enc1.attention_mask = [1, 1, 1] + [0] * 13
    enc2.ids = [4, 5] + [0] * 14
    enc2.attention_mask = [1, 1] + [0] * 14
    mock_tokenizer.encode_batch.return_value = [enc1, enc2]

    model = EmbeddingModel.__new__(EmbeddingModel)
    model.session = mock_session
    model.tokenizer = mock_tokenizer
    model._input_names = set()

    result = model.embed(["hello world", "def foo():"])
    assert result.shape == (2, 768)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


def test_reranker_returns_scores_between_0_and_1(tmp_path):
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)]

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode_batch.return_value = [MagicMock(ids=[1, 2], attention_mask=[1, 1])] * 3

    reranker = Reranker.__new__(Reranker)
    reranker.session = mock_session
    reranker.tokenizer = mock_tokenizer
    reranker._input_names = set()

    scores = reranker.rerank("query", ["a", "b", "c"])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
