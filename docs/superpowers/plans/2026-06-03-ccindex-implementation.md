# ccindex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ccindex`, a local offline-first CLI that indexes any codebase with ONNX-quantized code embeddings and injects semantically relevant chunks into AI coding agents (Claude Code, Gemini CLI, Antigravity) via their hook systems.

**Architecture:** Per-project SQLite + sqlite-vec stores code chunks embedded with a bundled ONNX jina-code model. Two-stage retrieval (ANN vector search → cross-encoder rerank) runs on every query, with git-commit-aware lazy incremental re-indexing before each search. Agent adapters auto-inject top-5 chunks before every AI response via each agent's hook system.

**Tech Stack:** Python ≥3.10, click, onnxruntime, tokenizers, sqlite-vec, tree-sitter ≥0.22 + language packages, rich, watchdog, pathspec, tomllib (stdlib 3.11+) / tomli (3.10 backport)

---

## File Structure

```
ccindex/
├── pyproject.toml
├── .gitattributes                    # Git LFS tracking for .onnx files
├── models/
│   ├── jina-code-onnx/              # Committed via Git LFS (~130MB)
│   │   ├── model.onnx
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── special_tokens_map.json
│   └── reranker-onnx/               # Committed via Git LFS (~85MB)
│       ├── model.onnx
│       └── tokenizer.json
├── scripts/
│   └── download_models.py           # One-time dev script to fetch + convert models
└── src/
    └── ccindex/
        ├── __init__.py              # version string
        ├── cli.py                   # All click CLI commands
        ├── config.py                # Load/merge user + project config.toml
        ├── models.py                # ONNX model loading, embed(), rerank(), path resolution
        ├── walker.py                # File discovery, .gitignore + .ccindexignore parsing
        ├── chunker.py               # tree-sitter code chunks + sliding window + Jupyter
        ├── index.py                 # SQLite + sqlite-vec schema, CRUD, FTS5, WAL
        ├── git.py                   # Branch/commit detection, diff, merge state
        ├── indexer.py               # Orchestrates walker→chunker→embed→index + progress
        ├── retrieval.py             # Lazy incremental check, vector+FTS search, rerank
        ├── daemon.py                # launchd/systemd/Windows Task Scheduler registration
        └── agents/
            ├── __init__.py
            ├── base.py              # Abstract AgentAdapter interface
            ├── claude_code.py       # Claude Code UserPromptSubmit hook
            ├── gemini_cli.py        # Gemini CLI hook
            └── antigravity.py      # Antigravity hook

tests/
├── conftest.py
├── test_config.py
├── test_models.py
├── test_walker.py
├── test_chunker.py
├── test_index.py
├── test_git.py
├── test_indexer.py
├── test_retrieval.py
├── test_agents.py
└── test_cli.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitattributes`
- Create: `src/ccindex/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ccindex"
version = "0.1.0"
description = "Local offline-first code indexing and semantic search for AI coding agents"
requires-python = ">=3.10"
dependencies = [
    "click>=8.1",
    "onnxruntime>=1.17",
    "tokenizers>=0.19",
    "sqlite-vec>=0.1.6",
    "tree-sitter>=0.22",
    "tree-sitter-python>=0.22",
    "tree-sitter-javascript>=0.22",
    "tree-sitter-typescript>=0.22",
    "tree-sitter-go>=0.22",
    "tree-sitter-rust>=0.22",
    "tree-sitter-java>=0.22",
    "tree-sitter-c>=0.22",
    "tree-sitter-cpp>=0.22",
    "tree-sitter-ruby>=0.22",
    "rich>=13.0",
    "watchdog>=4.0",
    "pathspec>=0.12",
    "tomli>=2.0; python_version < '3.11'",
]

[project.scripts]
ccindex = "ccindex.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
]

[tool.hatch.build.targets.wheel]
packages = ["src/ccindex"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitattributes` for Git LFS**

```
models/jina-code-onnx/*.onnx filter=lfs diff=lfs merge=lfs -text
models/reranker-onnx/*.onnx filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 3: Create `src/ccindex/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def tmp_project(tmp_path):
    """A temporary directory acting as a project root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    return tmp_path


@pytest.fixture
def tmp_git_project(tmp_project):
    """Temporary project with a git repo initialized."""
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_project, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_project, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_project, check=True, capture_output=True)
    return tmp_project
```

- [ ] **Step 5: Install and verify**

```bash
cd /Users/dillibabukadati/Documents/ccindex
pip install -e ".[dev]"
python -c "import ccindex; print(ccindex.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitattributes src/ tests/conftest.py
git commit -m "feat: project scaffold with dependencies and package structure"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `src/ccindex/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.config'`

- [ ] **Step 3: Implement `src/ccindex/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class Config:
    top_k: int = 5
    relevance_threshold: float = 0.3
    token_cap: int = 1500
    max_file_size_kb: int = 1024
    batch_size: int = 32
    max_stale_files: int = 200
    ignore_patterns: list[str] = field(default_factory=list)


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(project_root: Path | None = None) -> Config:
    user_cfg = _load_toml(Path.home() / ".ccindex" / "config.toml")
    proj_cfg = _load_toml(project_root / ".ccindex" / "config.toml") if project_root else {}

    merged = {}
    for section in ("query", "index", "ignore"):
        merged.update(user_cfg.get(section, {}))
        merged.update(proj_cfg.get(section, {}))

    return Config(
        top_k=merged.get("top_k", 5),
        relevance_threshold=merged.get("relevance_threshold", 0.3),
        token_cap=merged.get("token_cap", 1500),
        max_file_size_kb=merged.get("max_file_size_kb", 1024),
        batch_size=merged.get("batch_size", 32),
        max_stale_files=merged.get("max_stale_files", 200),
        ignore_patterns=merged.get("patterns", []),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/config.py tests/test_config.py
git commit -m "feat: configuration module with user and project-level config.toml"
```

---

## Task 3: ONNX Model Loading

**Files:**
- Create: `src/ccindex/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
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

    scores = reranker.rerank("query", ["a", "b", "c"])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.models'`

- [ ] **Step 3: Implement `src/ccindex/models.py`**

```python
from __future__ import annotations
from pathlib import Path
import struct
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_PACKAGE_ROOT = Path(__file__).parent.parent.parent
_USER_CACHE = Path.home() / ".ccindex" / "models"


class ModelNotFoundError(Exception):
    pass


def get_model_dir(model_name: str) -> Path:
    bundled = _PACKAGE_ROOT / "models" / model_name
    if bundled.exists():
        return bundled
    user_cached = _USER_CACHE / model_name
    if user_cached.exists():
        return user_cached
    raise ModelNotFoundError(
        f"Model '{model_name}' not found.\n"
        f"Run: ccindex update"
    )


class EmbeddingModel:
    def __init__(self, model_dir: Path):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, feed)
        hidden = outputs[0]  # (batch, seq_len, hidden_dim)

        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype(np.float32)


class Reranker:
    def __init__(self, model_dir: Path):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        pairs = [f"{query}[SEP]{p}" for p in passages]
        encoded = self.tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, feed)
        logits = outputs[0]
        scores = logits[:, 1] if logits.shape[1] == 2 else logits[:, 0]
        return (1.0 / (1.0 + np.exp(-scores))).tolist()


def serialize_embedding(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/models.py tests/test_models.py
git commit -m "feat: ONNX model loading for embedding and reranking"
```

---

## Task 4: File Walker

**Files:**
- Create: `src/ccindex/walker.py`
- Create: `tests/test_walker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_walker.py
import pytest
from pathlib import Path
from ccindex.config import Config
from ccindex.walker import walk_project


def _write(path: Path, content: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_discovers_python_files(tmp_path):
    _write(tmp_path / "src" / "main.py", "def foo(): pass")
    files = list(walk_project(tmp_path, Config()))
    assert any(f.name == "main.py" for f in files)


def test_skips_node_modules(tmp_path):
    _write(tmp_path / "node_modules" / "lodash" / "index.js")
    files = list(walk_project(tmp_path, Config()))
    assert not any("node_modules" in str(f) for f in files)


def test_skips_virtual_envs(tmp_path):
    _write(tmp_path / ".venv" / "lib" / "site.py")
    files = list(walk_project(tmp_path, Config()))
    assert not any(".venv" in str(f) for f in files)


def test_skips_lock_files(tmp_path):
    _write(tmp_path / "package-lock.json", '{}')
    _write(tmp_path / "poetry.lock", "# lock")
    files = list(walk_project(tmp_path, Config()))
    assert not any(f.name in ("package-lock.json", "poetry.lock") for f in files)


def test_skips_env_files(tmp_path):
    _write(tmp_path / ".env", "SECRET=abc")
    _write(tmp_path / ".env.local", "SECRET=def")
    files = list(walk_project(tmp_path, Config()))
    assert not any(f.name.startswith(".env") for f in files)


def test_respects_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("dist/\n")
    _write(tmp_path / "dist" / "bundle.js")
    _write(tmp_path / "src" / "app.js")
    files = list(walk_project(tmp_path, Config()))
    assert not any("dist" in str(f) for f in files)
    assert any(f.name == "app.js" for f in files)


def test_respects_ccindexignore(tmp_path):
    (tmp_path / ".ccindexignore").write_text("migrations/\n")
    _write(tmp_path / "migrations" / "001_init.sql")
    _write(tmp_path / "src" / "app.py")
    files = list(walk_project(tmp_path, Config()))
    assert not any("migrations" in str(f) for f in files)


def test_skips_files_over_size_limit(tmp_path):
    big_file = tmp_path / "big.py"
    big_file.write_text("x" * (1025 * 1024))  # 1025KB > 1024KB limit
    files = list(walk_project(tmp_path, Config()))
    assert big_file not in files


def test_skips_ccindex_dir(tmp_path):
    _write(tmp_path / ".ccindex" / "index.db")
    files = list(walk_project(tmp_path, Config()))
    assert not any(".ccindex" in str(f) for f in files)


def test_skips_minified_js(tmp_path):
    _write(tmp_path / "app.min.js", "var x=1;")
    files = list(walk_project(tmp_path, Config()))
    assert not any(f.name == "app.min.js" for f in files)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_walker.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.walker'`

- [ ] **Step 3: Implement `src/ccindex/walker.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import pathspec
from ccindex.config import Config

_ALWAYS_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "dist", "build",
    ".ccindex", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "coverage", ".coverage",
})

_LOCK_FILES = frozenset({
    "package-lock.json", "yarn.lock", "poetry.lock",
    "uv.lock", "Pipfile.lock", "composer.lock",
})

_SKIP_EXTENSIONS = frozenset({
    ".min.js", ".min.css",
})

_SKIP_SUFFIXES = frozenset({
    "_pb2.py", ".pb.go", ".generated.ts",
})

_SECRET_PATTERNS = frozenset({
    ".env", ".pem", ".key",
})

_TEXT_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql", ".md",
    ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ipynb",
    ".dockerfile", ".tf", ".hcl",
})


def _load_spec(root: Path, filename: str) -> pathspec.PathSpec | None:
    path = root / filename
    if not path.exists():
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", path.read_text().splitlines())


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" in chunk
    except OSError:
        return True


def _get_tsconfig_out_dir(root: Path) -> str | None:
    tsconfig = root / "tsconfig.json"
    if not tsconfig.exists():
        return None
    try:
        import json
        data = json.loads(tsconfig.read_text())
        return data.get("compilerOptions", {}).get("outDir")
    except (json.JSONDecodeError, OSError):
        return None


def walk_project(root: Path, config: Config) -> Iterator[Path]:
    gitignore = _load_spec(root, ".gitignore")
    ccignore = _load_spec(root, ".ccindexignore")
    ts_out_dir = _get_tsconfig_out_dir(root)  # e.g. "./dist" or "build"
    extra_spec = (
        pathspec.PathSpec.from_lines("gitwildmatch", config.ignore_patterns)
        if config.ignore_patterns else None
    )
    max_bytes = config.max_file_size_kb * 1024

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        rel_str = rel.as_posix()

        # Skip always-ignored dirs
        if any(part in _ALWAYS_SKIP_DIRS for part in rel.parts):
            continue

        # Skip TypeScript compiled output dir
        if ts_out_dir:
            out_dir_clean = ts_out_dir.lstrip("./")
            if rel.parts and rel.parts[0] == out_dir_clean:
                continue

        # Skip .gitignore'd
        if gitignore and gitignore.match_file(rel_str):
            continue

        # Skip .ccindexignore'd
        if ccignore and ccignore.match_file(rel_str):
            continue

        # Skip user config patterns
        if extra_spec and extra_spec.match_file(rel_str):
            continue

        # Skip lock files
        if path.name in _LOCK_FILES:
            continue

        # Skip secret files
        if path.name.startswith(".env") or any(path.name.endswith(s) for s in _SECRET_PATTERNS):
            continue

        # Skip minified/generated
        name_lower = path.name.lower()
        if any(name_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue
        if any(path.name.endswith(s) for s in _SKIP_SUFFIXES):
            continue

        # Skip unknown/binary extensions
        if path.suffix.lower() not in _TEXT_EXTENSIONS and path.suffix != "":
            continue

        # Skip oversized files
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue

        # Skip binary files
        if _is_binary(path):
            continue

        yield path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_walker.py -v
```

Expected: 10 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/walker.py tests/test_walker.py
git commit -m "feat: file walker with gitignore, ccindexignore, and skip rules"
```

---

## Task 5: Code Chunker (tree-sitter)

**Files:**
- Create: `src/ccindex/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chunker.py
import pytest
import json
from pathlib import Path
from ccindex.config import Config
from ccindex.chunker import chunk_file, Chunk


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_python_function_extracted_as_chunk(tmp_path):
    src = _write(tmp_path, "foo.py", "def greet(name):\n    return f'Hello {name}'\n")
    chunks = chunk_file(src, tmp_path, Config())
    assert len(chunks) == 1
    assert chunks[0].symbol == "greet"
    assert chunks[0].lang == "python"
    assert "def greet" in chunks[0].chunk_text
    assert chunks[0].file_path == "foo.py"
    assert chunks[0].start_line == 1


def test_python_class_extracted_as_chunk(tmp_path):
    src = _write(tmp_path, "model.py", "class User:\n    def __init__(self):\n        pass\n")
    chunks = chunk_file(src, tmp_path, Config())
    symbols = [c.symbol for c in chunks]
    assert "User" in symbols


def test_long_function_split_with_overlap(tmp_path):
    lines = ["def big():\n"] + [f"    x{i} = {i}\n" for i in range(150)]
    src = _write(tmp_path, "big.py", "".join(lines))
    chunks = chunk_file(src, tmp_path, Config())
    assert len(chunks) > 1


def test_markdown_sliding_window(tmp_path):
    content = "# Title\n\n" + "word " * 600
    src = _write(tmp_path, "README.md", content)
    chunks = chunk_file(src, tmp_path, Config())
    assert len(chunks) >= 1
    assert chunks[0].lang == "markdown"
    assert "# Title" in chunks[0].chunk_text


def test_small_json_as_single_chunk(tmp_path):
    src = _write(tmp_path, "config.json", '{"key": "value"}')
    chunks = chunk_file(src, tmp_path, Config())
    assert len(chunks) == 1
    assert chunks[0].lang == "json"


def test_jupyter_extracts_code_cells_only(tmp_path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title"]},
            {"cell_type": "code", "source": ["def foo():\n", "    pass"]},
            {"cell_type": "code", "source": ["x = 1"]},
        ]
    }
    src = _write(tmp_path, "notebook.ipynb", json.dumps(nb))
    chunks = chunk_file(src, tmp_path, Config())
    assert len(chunks) == 2
    assert all(c.lang == "python" for c in chunks)
    assert not any("# Title" in c.chunk_text for c in chunks)


def test_unknown_extension_falls_back_to_sliding_window(tmp_path):
    src = _write(tmp_path, "script.fish", "echo hello\n" * 50)
    chunks = chunk_file(src, tmp_path, Config())
    assert len(chunks) >= 1


def test_chunk_has_file_mtime(tmp_path):
    src = _write(tmp_path, "app.py", "def main(): pass\n")
    chunks = chunk_file(src, tmp_path, Config())
    assert all(c.file_mtime > 0 for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.chunker'`

- [ ] **Step 3: Implement `src/ccindex/chunker.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from ccindex.config import Config

_CHARS_PER_TOKEN = 4
_SLIDING_WINDOW_TOKENS = 512
_SLIDING_OVERLAP_TOKENS = 64
_CONFIG_WINDOW_TOKENS = 256
_CONFIG_OVERLAP_TOKENS = 32
_MAX_FUNCTION_LINES = 100
_FUNCTION_OVERLAP_LINES = 20


@dataclass
class Chunk:
    file_path: str
    start_line: int | None
    end_line: int | None
    symbol: str | None
    lang: str
    chunk_text: str
    file_mtime: float


_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust", ".java": "java", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".rb": "ruby",
    ".md": "markdown", ".txt": "text", ".rst": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".sql": "sql",
}

_CODE_LANGS = frozenset({"python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "ruby"})
_DOC_LANGS = frozenset({"markdown", "text"})
_CONFIG_LANGS = frozenset({"json", "yaml", "toml", "sql"})
_CONFIG_MAX_BYTES = 2 * 1024


def _get_parser(lang: str):
    try:
        from tree_sitter import Language, Parser
        if lang == "python":
            import tree_sitter_python as ts_lang
        elif lang in ("javascript",):
            import tree_sitter_javascript as ts_lang
        elif lang in ("typescript",):
            import tree_sitter_typescript
            ts_lang = tree_sitter_typescript.language_typescript()
            return Parser(ts_lang)
        elif lang == "go":
            import tree_sitter_go as ts_lang
        elif lang == "rust":
            import tree_sitter_rust as ts_lang
        elif lang == "java":
            import tree_sitter_java as ts_lang
        elif lang == "c":
            import tree_sitter_c as ts_lang
        elif lang == "cpp":
            import tree_sitter_cpp as ts_lang
        elif lang == "ruby":
            import tree_sitter_ruby as ts_lang
        else:
            return None
        return Parser(Language(ts_lang.language()))
    except (ImportError, Exception):
        return None


_SYMBOL_QUERIES = {
    "python": """
        (function_definition name: (identifier) @name) @node
        (class_definition name: (identifier) @name) @node
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @node
        (class_declaration name: (identifier) @name) @node
        (method_definition name: (property_identifier) @name) @node
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @node
        (class_declaration name: (identifier) @name) @node
        (method_definition name: (property_identifier) @name) @node
    """,
    "go": """
        (function_declaration name: (identifier) @name) @node
        (method_declaration name: (field_identifier) @name) @node
    """,
    "rust": """
        (function_item name: (identifier) @name) @node
        (impl_item) @node
    """,
    "java": """
        (method_declaration name: (identifier) @name) @node
        (class_declaration name: (identifier) @name) @node
    """,
}


def _treesitter_chunks(path: Path, rel: str, lang: str, source: str, mtime: float) -> list[Chunk]:
    parser = _get_parser(lang)
    if parser is None:
        return []

    tree = parser.parse(bytes(source, "utf-8"))
    lines = source.splitlines()
    chunks: list[Chunk] = []
    query_str = _SYMBOL_QUERIES.get(lang, "")
    if not query_str:
        return []

    try:
        from tree_sitter import Language
        if lang == "python":
            import tree_sitter_python as ts_lang
            language = Language(ts_lang.language())
        elif lang == "javascript":
            import tree_sitter_javascript as ts_lang
            language = Language(ts_lang.language())
        elif lang == "go":
            import tree_sitter_go as ts_lang
            language = Language(ts_lang.language())
        elif lang == "rust":
            import tree_sitter_rust as ts_lang
            language = Language(ts_lang.language())
        elif lang == "java":
            import tree_sitter_java as ts_lang
            language = Language(ts_lang.language())
        else:
            return []

        query = language.query(query_str)
        captures = query.captures(tree.root_node)
    except Exception:
        return []

    seen_nodes = set()
    for node, capture_name in captures:
        if capture_name != "node" or id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_lines = lines[node.start_point[0]:node.end_point[0] + 1]

        symbol = None
        for child_node, child_name in captures:
            if child_name == "name" and node.start_point[0] <= child_node.start_point[0] <= node.end_point[0]:
                symbol = source[child_node.start_byte:child_node.end_byte]
                break

        if len(node_lines) <= _MAX_FUNCTION_LINES:
            chunks.append(Chunk(
                file_path=rel,
                start_line=start_line,
                end_line=end_line,
                symbol=symbol,
                lang=lang,
                chunk_text="\n".join(node_lines),
                file_mtime=mtime,
            ))
        else:
            step = _MAX_FUNCTION_LINES - _FUNCTION_OVERLAP_LINES
            for i in range(0, len(node_lines), step):
                slice_lines = node_lines[i:i + _MAX_FUNCTION_LINES]
                if not slice_lines:
                    break
                chunks.append(Chunk(
                    file_path=rel,
                    start_line=start_line + i,
                    end_line=start_line + i + len(slice_lines) - 1,
                    symbol=symbol,
                    lang=lang,
                    chunk_text="\n".join(slice_lines),
                    file_mtime=mtime,
                ))

    return chunks


def _sliding_window_chunks(
    path: Path, rel: str, lang: str, source: str, mtime: float,
    window_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    window_chars = window_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    pos = 0
    line_starts = []
    char = 0
    for line in lines:
        line_starts.append(char)
        char += len(line)
    text = source

    while pos < len(text):
        end = min(pos + window_chars, len(text))
        slice_text = text[pos:end]

        start_line = next((i + 1 for i, ls in enumerate(line_starts) if ls >= pos), 1)
        end_line = next((i + 1 for i, ls in enumerate(line_starts) if ls >= end), len(lines))

        # Prepend heading for markdown
        prefix = ""
        if lang == "markdown":
            heading_lines = [l.strip() for l in slice_text.splitlines() if l.startswith("#")]
            if not heading_lines:
                prev_headings = [l.strip() for l in text[:pos].splitlines() if l.startswith("#")]
                if prev_headings:
                    prefix = prev_headings[-1] + "\n"

        chunks.append(Chunk(
            file_path=rel,
            start_line=start_line,
            end_line=end_line,
            symbol=None,
            lang=lang,
            chunk_text=prefix + slice_text,
            file_mtime=mtime,
        ))

        if end == len(text):
            break
        pos += window_chars - overlap_chars

    return chunks


def _jupyter_chunks(path: Path, rel: str, mtime: float) -> list[Chunk]:
    try:
        nb = json.loads(path.read_bytes())
    except (json.JSONDecodeError, OSError):
        return []

    chunks: list[Chunk] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        text = "".join(source_lines).strip()
        if not text:
            continue
        chunks.append(Chunk(
            file_path=rel,
            start_line=None,
            end_line=None,
            symbol=f"cell_{i}",
            lang="python",
            chunk_text=text,
            file_mtime=mtime,
        ))
    return chunks


def chunk_file(path: Path, root: Path, config: Config) -> list[Chunk]:
    rel = path.relative_to(root).as_posix()
    mtime = path.stat().st_mtime
    ext = path.suffix.lower()

    if ext == ".ipynb":
        return _jupyter_chunks(path, rel, mtime)

    lang = _EXT_TO_LANG.get(ext, "text")

    try:
        source = path.read_text(errors="replace")
    except OSError:
        return []

    if lang in _CODE_LANGS:
        chunks = _treesitter_chunks(path, rel, lang, source, mtime)
        if chunks:
            return chunks
        return _sliding_window_chunks(path, rel, lang, source, mtime, 128, 32)

    if lang in _DOC_LANGS:
        return _sliding_window_chunks(path, rel, lang, source, mtime, _SLIDING_WINDOW_TOKENS, _SLIDING_OVERLAP_TOKENS)

    if lang in _CONFIG_LANGS:
        if len(source.encode()) <= _CONFIG_MAX_BYTES:
            return [Chunk(file_path=rel, start_line=1, end_line=source.count("\n") + 1,
                          symbol=None, lang=lang, chunk_text=source, file_mtime=mtime)]
        return _sliding_window_chunks(path, rel, lang, source, mtime, _CONFIG_WINDOW_TOKENS, _CONFIG_OVERLAP_TOKENS)

    return _sliding_window_chunks(path, rel, lang, source, mtime, 128, 32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/chunker.py tests/test_chunker.py
git commit -m "feat: chunker with tree-sitter code extraction and sliding window fallback"
```

---

## Task 6: SQLite Index Storage

**Files:**
- Create: `src/ccindex/index.py`
- Create: `tests/test_index.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_index.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.index'`

- [ ] **Step 3: Implement `src/ccindex/index.py`**

```python
from __future__ import annotations
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sqlite_vec
from ccindex.chunker import Chunk


@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    start_line: int | None
    end_line: int | None
    symbol: str | None
    lang: str
    chunk_text: str
    score: float


def _serialize(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


class Index:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self._check_schema_version()

    _SCHEMA_VERSION = "1"

    def _check_schema_version(self):
        stored = self.get_meta("schema_version")
        if stored is None:
            self.set_meta("schema_version", self._SCHEMA_VERSION)
        elif stored != self._SCHEMA_VERSION:
            # Schema changed — drop and recreate
            self._conn.executescript("""
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunks_vec;
            """)
            self._create_schema()
            self.set_meta("schema_version", self._SCHEMA_VERSION)
            self.set_meta("index_state", "partial")

    def _create_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT NOT NULL,
                start_line  INTEGER,
                end_line    INTEGER,
                symbol      TEXT,
                lang        TEXT,
                chunk_text  TEXT NOT NULL,
                file_mtime  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_text, symbol, file_path,
                content='chunks', content_rowid='id',
                tokenize='porter ascii'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding FLOAT[768]
            );
        """)
        self._conn.commit()

    def upsert_chunks(self, items: list[tuple[Chunk, np.ndarray]]):
        if not items:
            return
        paths = {chunk.file_path for chunk, _ in items}
        for path in paths:
            self.delete_by_path(path)

        with self._conn:
            for chunk, embedding in items:
                cur = self._conn.execute(
                    """INSERT INTO chunks
                       (file_path, start_line, end_line, symbol, lang, chunk_text, file_mtime)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk.file_path, chunk.start_line, chunk.end_line,
                     chunk.symbol, chunk.lang, chunk.chunk_text, chunk.file_mtime),
                )
                row_id = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    (row_id, _serialize(embedding)),
                )
                self._conn.execute(
                    "INSERT INTO chunks_fts(rowid, chunk_text, symbol, file_path) VALUES (?, ?, ?, ?)",
                    (row_id, chunk.chunk_text, chunk.symbol or "", chunk.file_path),
                )

    def delete_by_path(self, file_path: str):
        ids = [r[0] for r in self._conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._conn:
            self._conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", ids)
            self._conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", ids)
            self._conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)

    def get_all_mtimes(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT file_path, MAX(file_mtime) FROM chunks GROUP BY file_path"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def vector_search(self, embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        rows = self._conn.execute(
            """SELECT c.id, c.file_path, c.start_line, c.end_line, c.symbol,
                      c.lang, c.chunk_text, v.distance
               FROM chunks_vec v
               JOIN chunks c ON c.id = v.rowid
               WHERE v.embedding MATCH ?
               ORDER BY v.distance
               LIMIT ?""",
            (_serialize(embedding), top_k),
        ).fetchall()
        return [SearchResult(
            chunk_id=r[0], file_path=r[1], start_line=r[2], end_line=r[3],
            symbol=r[4], lang=r[5], chunk_text=r[6],
            score=1.0 - float(r[7]),
        ) for r in rows]

    def fts_search(self, query: str, top_k: int) -> list[SearchResult]:
        safe_query = query.replace('"', '""')
        rows = self._conn.execute(
            """SELECT c.id, c.file_path, c.start_line, c.end_line, c.symbol,
                      c.lang, c.chunk_text, rank
               FROM chunks_fts
               JOIN chunks c ON c.id = chunks_fts.rowid
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, top_k),
        ).fetchall()
        return [SearchResult(
            chunk_id=r[0], file_path=r[1], start_line=r[2], end_line=r[3],
            symbol=r[4], lang=r[5], chunk_text=r[6], score=0.5,
        ) for r in rows]

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str):
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_index.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/index.py tests/test_index.py
git commit -m "feat: SQLite + sqlite-vec index with WAL, FTS5, and vector search"
```

---

## Task 7: Git Integration

**Files:**
- Create: `src/ccindex/git.py`
- Create: `tests/test_git.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_git.py
import pytest
import subprocess
from pathlib import Path
from ccindex.git import get_current_commit, get_changed_files, is_merge_in_progress, find_repo_root


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True)


def _commit(repo, msg="init"):
    _git(["config", "user.email", "t@t.com"], repo)
    _git(["config", "user.name", "T"], repo)
    (repo / "file.txt").write_text(msg)
    _git(["add", "."], repo)
    _git(["commit", "-m", msg], repo)


def test_get_current_commit_in_repo(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path)
    commit = get_current_commit(tmp_path)
    assert commit is not None and len(commit) == 40


def test_get_current_commit_outside_repo(tmp_path):
    assert get_current_commit(tmp_path) is None


def test_get_changed_files_between_commits(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path, "first")
    first = get_current_commit(tmp_path)

    (tmp_path / "new_file.py").write_text("x = 1")
    _git(["add", "."], tmp_path)
    _git(["commit", "-m", "second"], tmp_path)

    changed = get_changed_files(tmp_path, first, "HEAD")
    assert "new_file.py" in changed


def test_get_changed_files_returns_empty_for_same_commit(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path)
    commit = get_current_commit(tmp_path)
    changed = get_changed_files(tmp_path, commit, commit)
    assert changed == []


def test_is_merge_in_progress_false_normally(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path)
    assert is_merge_in_progress(tmp_path) is False


def test_find_repo_root_from_subdir(tmp_path):
    _git(["init"], tmp_path)
    subdir = tmp_path / "src" / "deep"
    subdir.mkdir(parents=True)
    root = find_repo_root(subdir)
    assert root == tmp_path


def test_find_repo_root_returns_none_outside_repo(tmp_path):
    assert find_repo_root(tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_git.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.git'`

- [ ] **Step 3: Implement `src/ccindex/git.py`**

```python
from __future__ import annotations
import subprocess
from pathlib import Path


def _run(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def find_repo_root(start: Path) -> Path | None:
    out = _run(["git", "rev-parse", "--show-toplevel"], start)
    return Path(out) if out else None


def get_current_commit(repo_root: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], repo_root)


def get_current_branch(repo_root: Path) -> str | None:
    return _run(["git", "branch", "--show-current"], repo_root)


def get_changed_files(repo_root: Path, from_commit: str, to_commit: str) -> list[str]:
    if from_commit == to_commit:
        return []
    out = _run(
        ["git", "diff", "--name-only", from_commit, to_commit],
        repo_root,
    )
    if not out:
        return []
    return [f for f in out.splitlines() if f]


def is_merge_in_progress(repo_root: Path) -> bool:
    return (repo_root / ".git" / "MERGE_HEAD").exists()


def install_post_checkout_hook(repo_root: Path) -> bool:
    hook_path = repo_root / ".git" / "hooks" / "post-checkout"
    script = "#!/bin/sh\nccindex index\n"
    try:
        hook_path.write_text(script)
        hook_path.chmod(0o755)
        return True
    except OSError:
        return False


def install_post_merge_hook(repo_root: Path) -> bool:
    hook_path = repo_root / ".git" / "hooks" / "post-merge"
    script = "#!/bin/sh\nccindex index\n"
    try:
        hook_path.write_text(script)
        hook_path.chmod(0o755)
        return True
    except OSError:
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_git.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/git.py tests/test_git.py
git commit -m "feat: git integration for branch/commit detection and hook installation"
```

---

## Task 8: Indexer Orchestration

**Files:**
- Create: `src/ccindex/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_indexer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.indexer'`

- [ ] **Step 3: Implement `src/ccindex/indexer.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
from ccindex.config import Config
from ccindex.walker import walk_project
from ccindex.chunker import chunk_file, Chunk
from ccindex.index import Index
from ccindex.models import EmbeddingModel


class Indexer:
    def __init__(
        self,
        root: Path,
        config: Config,
        model: EmbeddingModel,
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        self.root = root
        self.config = config
        self.model = model
        self.progress_cb = progress_cb
        self._db_path = root / ".ccindex" / "index.db"
        self._index = Index(self._db_path)

    @property
    def index(self) -> Index:
        return self._index

    def _ensure_gitignore(self):
        gi = self.root / ".gitignore"
        entry = ".ccindex/\n"
        if gi.exists():
            if ".ccindex/" not in gi.read_text():
                with open(gi, "a") as f:
                    f.write(entry)
        else:
            gi.write_text(entry)

    def _embed_batch(self, chunks: list[Chunk]) -> list[tuple[Chunk, np.ndarray]]:
        batch_size = self.config.batch_size
        results: list[tuple[Chunk, np.ndarray]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.chunk_text for c in batch]
            embeddings = self.model.embed(texts)
            for chunk, emb in zip(batch, embeddings):
                results.append((chunk, emb))
        return results

    def run_full(self, show_progress: bool = False):
        self._ensure_gitignore()
        self._index.set_meta("index_state", "partial")

        files = list(walk_project(self.root, self.config))
        total = len(files)

        for i, path in enumerate(files):
            chunks = chunk_file(path, self.root, self.config)
            if not chunks:
                continue
            items = self._embed_batch(chunks)
            self._index.upsert_chunks(items)
            if self.progress_cb:
                self.progress_cb(i + 1, total)

        self._index.set_meta("index_state", "complete")

    def run_incremental(self, changed_paths: list[str] | None = None):
        stored_mtimes = self._index.get_all_mtimes()
        current_files = {
            p.relative_to(self.root).as_posix(): p
            for p in walk_project(self.root, self.config)
        }

        if changed_paths is not None:
            to_update = [
                current_files[p] for p in changed_paths
                if p in current_files
            ]
        else:
            to_update = []
            for rel, path in current_files.items():
                current_mtime = path.stat().st_mtime
                if rel not in stored_mtimes or stored_mtimes[rel] != current_mtime:
                    to_update.append(path)

        # Remove deleted files
        current_rels = set(current_files.keys())
        for stored_rel in list(stored_mtimes.keys()):
            if stored_rel not in current_rels:
                self._index.delete_by_path(stored_rel)

        for path in to_update:
            chunks = chunk_file(path, self.root, self.config)
            if not chunks:
                self._index.delete_by_path(path.relative_to(self.root).as_posix())
                continue
            items = self._embed_batch(chunks)
            self._index.upsert_chunks(items)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_indexer.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/indexer.py tests/test_indexer.py
git commit -m "feat: indexer orchestration with full and incremental re-indexing"
```

---

## Task 9: Retrieval Pipeline

**Files:**
- Create: `src/ccindex/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retrieval.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from ccindex.config import Config
from ccindex.index import Index, SearchResult
from ccindex.retrieval import Retriever, format_hook_output


def _make_result(symbol="foo", file_path="src/foo.py", score=0.8):
    return SearchResult(
        chunk_id=1, file_path=file_path,
        start_line=1, end_line=10,
        symbol=symbol, lang="python",
        chunk_text=f"def {symbol}(): pass",
        score=score,
    )


def _make_retriever(tmp_path, top_k=5, threshold=0.3):
    cfg = Config(top_k=top_k, relevance_threshold=threshold, token_cap=1500)
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
    candidates = [_make_result(symbol=f"fn{i}", score=0.8) for i in range(3)]
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.retrieval'`

- [ ] **Step 3: Implement `src/ccindex/retrieval.py`**

```python
from __future__ import annotations
import numpy as np
from ccindex.config import Config
from ccindex.index import Index, SearchResult
from ccindex.models import EmbeddingModel, Reranker

_CHARS_PER_TOKEN = 4


class Retriever:
    def __init__(
        self,
        index: Index,
        model: EmbeddingModel,
        reranker: Reranker,
        config: Config,
    ):
        self._index = index
        self._model = model
        self._reranker = reranker
        self._config = config

    def query(self, text: str) -> list[SearchResult]:
        query_vec = self._model.embed([text])[0]

        vec_results = self._index.vector_search(query_vec, top_k=50)
        fts_results = self._index.fts_search(text, top_k=20)
        candidates = self._dedupe(vec_results + fts_results)

        if not candidates:
            return []

        passages = [r.chunk_text for r in candidates]
        scores = self._reranker.rerank(text, passages)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        filtered = [
            result for result, score in ranked
            if score >= self._config.relevance_threshold
        ]

        capped = self._apply_token_cap(filtered)
        return capped[:self._config.top_k]

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        seen: set[int] = set()
        out: list[SearchResult] = []
        for r in results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                out.append(r)
        return out

    def _apply_token_cap(self, results: list[SearchResult]) -> list[SearchResult]:
        cap_chars = self._config.token_cap * _CHARS_PER_TOKEN
        used = 0
        out: list[SearchResult] = []
        for r in results:
            size = len(r.chunk_text)
            if used + size > cap_chars:
                break
            out.append(r)
            used += size
        return out


def format_hook_output(results: list[SearchResult]) -> str:
    if not results:
        return ""

    lines = ["[ccindex context]"]
    for r in results:
        loc = f"{r.file_path}"
        if r.start_line and r.end_line:
            loc += f":{r.start_line}-{r.end_line}"
        if r.symbol:
            loc += f" ({r.symbol})"
        lines.append(f"── {loc}")
        for line in r.chunk_text.splitlines()[:20]:
            lines.append(f"   {line}")
        lines.append("")
    lines.append("[end ccindex context]")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/retrieval.py tests/test_retrieval.py
git commit -m "feat: two-stage retrieval pipeline with rerank, threshold, and token cap"
```

---

## Task 10: Agent Adapters

**Files:**
- Create: `src/ccindex/agents/__init__.py`
- Create: `src/ccindex/agents/base.py`
- Create: `src/ccindex/agents/claude_code.py`
- Create: `src/ccindex/agents/gemini_cli.py`
- Create: `src/ccindex/agents/antigravity.py`
- Create: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents.py
import pytest
import json
from pathlib import Path
from ccindex.agents.claude_code import ClaudeCodeAdapter
from ccindex.agents.gemini_cli import GeminiCliAdapter


def test_claude_code_install_writes_hook(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    assert settings_path.exists()
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    assert any("ccindex" in h["command"] for h in hooks)


def test_claude_code_install_idempotent(tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    adapter.install()
    settings_path = tmp_path / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    ccindex_hooks = [h for h in hooks if "ccindex" in h["command"]]
    assert len(ccindex_hooks) == 1


def test_claude_code_install_preserves_existing_hooks(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    existing = {
        "hooks": {
            "UserPromptSubmit": [{
                "matcher": ".*",
                "hooks": [{"type": "command", "command": "other-tool"}]
            }]
        }
    }
    settings_path.write_text(json.dumps(existing))
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    commands = [h["command"] for h in hooks]
    assert "other-tool" in commands
    assert any("ccindex" in c for c in commands)


def test_claude_code_uninstall_removes_hook(tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    adapter.uninstall()
    settings_path = tmp_path / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    hooks = data.get("hooks", {}).get("UserPromptSubmit", [{}])[0].get("hooks", [])
    assert not any("ccindex" in h.get("command", "") for h in hooks)


def test_claude_code_check_returns_true_after_install(tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    assert not adapter.is_installed()
    adapter.install()
    assert adapter.is_installed()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agents.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement agent adapter files**

`src/ccindex/agents/__init__.py`:
```python
from ccindex.agents.base import AgentAdapter
from ccindex.agents.claude_code import ClaudeCodeAdapter
from ccindex.agents.gemini_cli import GeminiCliAdapter
from ccindex.agents.antigravity import AntigravityAdapter

ADAPTERS: dict[str, type[AgentAdapter]] = {
    "claude-code": ClaudeCodeAdapter,
    "gemini-cli": GeminiCliAdapter,
    "antigravity": AntigravityAdapter,
}
```

`src/ccindex/agents/base.py`:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class AgentAdapter(ABC):
    def __init__(self, project_root: Path):
        self.project_root = project_root

    @abstractmethod
    def install(self) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...

    @abstractmethod
    def is_installed(self) -> bool: ...
```

`src/ccindex/agents/claude_code.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from ccindex.agents.base import AgentAdapter

_HOOK_COMMAND = "ccindex query --top 5 --format hook"
_TIMEOUT_MS = 3000


class ClaudeCodeAdapter(AgentAdapter):
    @property
    def _settings_path(self) -> Path:
        return self.project_root / ".claude" / "settings.json"

    def _load_settings(self) -> dict:
        if self._settings_path.exists():
            try:
                return json.loads(self._settings_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_settings(self, data: dict):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(data, indent=2))

    def install(self):
        data = self._load_settings()
        data.setdefault("hooks", {})
        data["hooks"].setdefault("UserPromptSubmit", [{"matcher": ".*", "hooks": []}])

        entry = data["hooks"]["UserPromptSubmit"][0]
        entry.setdefault("hooks", [])

        if not any(_HOOK_COMMAND in h.get("command", "") for h in entry["hooks"]):
            entry["hooks"].append({
                "type": "command",
                "command": _HOOK_COMMAND,
                "timeout": _TIMEOUT_MS,
            })

        self._save_settings(data)

    def uninstall(self):
        data = self._load_settings()
        try:
            hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
            data["hooks"]["UserPromptSubmit"][0]["hooks"] = [
                h for h in hooks if _HOOK_COMMAND not in h.get("command", "")
            ]
            self._save_settings(data)
        except (KeyError, IndexError):
            pass

    def is_installed(self) -> bool:
        data = self._load_settings()
        try:
            hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
            return any(_HOOK_COMMAND in h.get("command", "") for h in hooks)
        except (KeyError, IndexError):
            return False
```

`src/ccindex/agents/gemini_cli.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from ccindex.agents.base import AgentAdapter

_HOOK_COMMAND = "ccindex query --top 5 --format hook"


class GeminiCliAdapter(AgentAdapter):
    @property
    def _settings_path(self) -> Path:
        return self.project_root / ".gemini" / "settings.json"

    def install(self):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self._settings_path.exists():
            try:
                data = json.loads(self._settings_path.read_text())
            except json.JSONDecodeError:
                pass
        data.setdefault("hooks", {})
        data["hooks"].setdefault("userPromptSubmit", [])
        if _HOOK_COMMAND not in data["hooks"]["userPromptSubmit"]:
            data["hooks"]["userPromptSubmit"].append(_HOOK_COMMAND)
        self._settings_path.write_text(json.dumps(data, indent=2))

    def uninstall(self):
        if not self._settings_path.exists():
            return
        try:
            data = json.loads(self._settings_path.read_text())
            cmds = data.get("hooks", {}).get("userPromptSubmit", [])
            data["hooks"]["userPromptSubmit"] = [c for c in cmds if c != _HOOK_COMMAND]
            self._settings_path.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, KeyError):
            pass

    def is_installed(self) -> bool:
        if not self._settings_path.exists():
            return False
        try:
            data = json.loads(self._settings_path.read_text())
            return _HOOK_COMMAND in data.get("hooks", {}).get("userPromptSubmit", [])
        except (json.JSONDecodeError, KeyError):
            return False
```

`src/ccindex/agents/antigravity.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from ccindex.agents.base import AgentAdapter

_HOOK_COMMAND = "ccindex query --top 5 --format hook"


class AntigravityAdapter(AgentAdapter):
    @property
    def _settings_path(self) -> Path:
        return self.project_root / ".antigravity" / "hooks.json"

    def install(self):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self._settings_path.exists():
            try:
                data = json.loads(self._settings_path.read_text())
            except json.JSONDecodeError:
                pass
        data.setdefault("on_prompt", [])
        if _HOOK_COMMAND not in data["on_prompt"]:
            data["on_prompt"].append(_HOOK_COMMAND)
        self._settings_path.write_text(json.dumps(data, indent=2))

    def uninstall(self):
        if not self._settings_path.exists():
            return
        try:
            data = json.loads(self._settings_path.read_text())
            data["on_prompt"] = [c for c in data.get("on_prompt", []) if c != _HOOK_COMMAND]
            self._settings_path.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, KeyError):
            pass

    def is_installed(self) -> bool:
        if not self._settings_path.exists():
            return False
        try:
            data = json.loads(self._settings_path.read_text())
            return _HOOK_COMMAND in data.get("on_prompt", [])
        except (json.JSONDecodeError, KeyError):
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agents.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ccindex/agents/ tests/test_agents.py
git commit -m "feat: agent adapters for Claude Code, Gemini CLI, and Antigravity"
```

---

## Task 11: CLI Commands

**Files:**
- Create: `src/ccindex/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
import pytest
import json
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from ccindex.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_status_no_index(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "No index found" in result.output


def test_clear_no_index(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["clear", "--yes"])
    assert result.exit_code == 0


def test_install_for_claude_code(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(main, ["install", "--for", "claude-code"])
    assert result.exit_code == 0


def test_install_unsupported_agent(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["install", "--for", "unknown-agent"])
    assert result.exit_code != 0
    assert "unknown-agent" in result.output


def test_query_no_index_exits_cleanly(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["query", "how does auth work"])
    assert result.exit_code == 1


def test_query_format_hook_outputs_empty_when_no_index(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["query", "auth", "--format", "hook"])
    assert result.exit_code == 1
    assert result.output.strip() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ccindex.cli'`

- [ ] **Step 3: Implement `src/ccindex/cli.py`**

```python
from __future__ import annotations
import sys
import json
from pathlib import Path
import click
from ccindex import __version__
from ccindex.config import load_config
from ccindex.agents import ADAPTERS


def _find_project_root() -> Path:
    """Walk up from CWD to find nearest .ccindex/ dir (monorepo support)."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".ccindex").exists():
            return parent
    return current  # fallback: treat CWD as root


def _get_db_path(root: Path) -> Path:
    return root / ".ccindex" / "index.db"


@click.group()
@click.version_option(__version__, prog_name="ccindex")
def main():
    """Local offline-first code indexer for AI coding agents."""
    pass


@main.command()
@click.option("--show-progress", is_flag=True, default=True)
def index(show_progress):
    """Index the current project (incremental if already indexed)."""
    from ccindex.models import get_model_dir, EmbeddingModel, ModelNotFoundError
    from ccindex.indexer import Indexer

    root = _find_project_root()
    config = load_config(root)
    db_path = _get_db_path(root)

    try:
        model_dir = get_model_dir("jina-code-onnx")
    except ModelNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(2)

    model = EmbeddingModel(model_dir)

    if show_progress:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Indexing...", total=None)

            def cb(done, total):
                progress.update(task, completed=done, total=total)

            indexer = Indexer(root=root, config=config, model=model, progress_cb=cb)
            if db_path.exists():
                from ccindex.index import Index
                idx = Index(db_path)
                state = idx.get_meta("index_state")
                stored_commit = idx.get_meta("git_commit_hash")

                from ccindex import git
                repo_root = git.find_repo_root(root)
                if repo_root and stored_commit:
                    current_commit = git.get_current_commit(repo_root)
                    if current_commit and current_commit != stored_commit:
                        if git.is_merge_in_progress(repo_root):
                            click.echo("Warning: merge in progress — skipping re-index until resolved.")
                            return
                        changed = git.get_changed_files(repo_root, stored_commit, current_commit)
                        if len(changed) <= config.max_stale_files:
                            indexer.run_incremental(changed_paths=changed)
                            idx.set_meta("git_commit_hash", current_commit)
                            idx.set_meta("git_branch", git.get_current_branch(repo_root) or "")
                        else:
                            click.echo(f"Warning: {len(changed)} files changed — index may be stale. Run `ccindex index` for a full refresh.")
                            return
                    else:
                        indexer.run_incremental()
                else:
                    indexer.run_incremental()
            else:
                click.echo("First run — building index...")
                indexer.run_full(show_progress=True)
                from ccindex import git
                repo_root = git.find_repo_root(root)
                if repo_root:
                    commit = git.get_current_commit(repo_root)
                    branch = git.get_current_branch(repo_root)
                    if commit:
                        indexer.index.set_meta("git_commit_hash", commit)
                    if branch:
                        indexer.index.set_meta("git_branch", branch)

        click.echo("Index updated.")
    else:
        indexer = Indexer(root=root, config=config, model=model)
        if db_path.exists():
            indexer.run_incremental()
        else:
            indexer.run_full()


@main.command()
@click.argument("text")
@click.option("--top", default=5, show_default=True)
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json", "hook"]))
def query(text, top, fmt):
    """Search the index for relevant code chunks."""
    from ccindex.models import get_model_dir, EmbeddingModel, Reranker, ModelNotFoundError
    from ccindex.index import Index
    from ccindex.retrieval import Retriever, format_hook_output

    root = _find_project_root()
    db_path = _get_db_path(root)
    config = load_config(root)
    config.top_k = top

    if not db_path.exists():
        if fmt == "hook":
            sys.exit(1)
        click.echo("No index found. Run: ccindex index", err=True)
        sys.exit(1)

    try:
        embed_dir = get_model_dir("jina-code-onnx")
        rerank_dir = get_model_dir("reranker-onnx")
    except ModelNotFoundError as e:
        if fmt == "hook":
            sys.exit(1)
        click.echo(str(e), err=True)
        sys.exit(2)

    # Lazy incremental re-index
    _lazy_reindex(root, config, embed_dir, db_path)

    model = EmbeddingModel(embed_dir)
    reranker = Reranker(rerank_dir)
    index = Index(db_path)
    retriever = Retriever(index=index, model=model, reranker=reranker, config=config)

    results = retriever.query(text)

    if fmt == "hook":
        output = format_hook_output(results)
        if output:
            click.echo(output)
        sys.exit(0 if results else 1)
    elif fmt == "json":
        click.echo(json.dumps([{
            "file": r.file_path, "start_line": r.start_line,
            "end_line": r.end_line, "symbol": r.symbol,
            "score": r.score, "text": r.chunk_text,
        } for r in results]))
    else:
        if not results:
            click.echo("No results found.")
            sys.exit(1)
        for r in results:
            loc = f"{r.file_path}"
            if r.start_line:
                loc += f":{r.start_line}"
            if r.symbol:
                loc += f" ({r.symbol})"
            click.echo(f"\n── {loc}")
            click.echo(r.chunk_text[:500])


def _lazy_reindex(root: Path, config, embed_dir: Path, db_path: Path):
    from ccindex.models import EmbeddingModel
    from ccindex.index import Index
    from ccindex.indexer import Indexer
    from ccindex import git

    try:
        idx = Index(db_path)
        stored_commit = idx.get_meta("git_commit_hash")
        repo_root = git.find_repo_root(root)

        if repo_root and stored_commit:
            current_commit = git.get_current_commit(repo_root)
            if current_commit and current_commit != stored_commit:
                if git.is_merge_in_progress(repo_root):
                    return
                changed = git.get_changed_files(repo_root, stored_commit, current_commit)
                if len(changed) <= config.max_stale_files:
                    model = EmbeddingModel(embed_dir)
                    indexer = Indexer(root=root, config=config, model=model)
                    indexer.run_incremental(changed_paths=changed)
                    idx.set_meta("git_commit_hash", current_commit)
                    idx.set_meta("git_branch", git.get_current_branch(repo_root) or "")
    except Exception:
        pass  # never block a query due to re-index failure


@main.command()
def status():
    """Show index statistics."""
    root = _find_project_root()
    db_path = _get_db_path(root)

    if not db_path.exists():
        click.echo("No index found. Run: ccindex index")
        return

    from ccindex.index import Index
    import os
    idx = Index(db_path)
    mtimes = idx.get_all_mtimes()
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    branch = idx.get_meta("git_branch") or "unknown"
    commit = (idx.get_meta("git_commit_hash") or "")[:8]

    click.echo(f"Files indexed : {len(mtimes)}")
    click.echo(f"Index size    : {size_mb:.1f} MB")
    click.echo(f"Branch        : {branch} @ {commit}")
    click.echo(f"State         : {idx.get_meta('index_state') or 'unknown'}")


@main.command()
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clear(yes):
    """Wipe the index database."""
    root = _find_project_root()
    db_path = _get_db_path(root)

    if not db_path.exists():
        click.echo("No index to clear.")
        return

    if not yes:
        click.confirm("This will delete the index. Continue?", abort=True)

    db_path.unlink()
    click.echo("Index cleared.")


@main.command()
@click.option("--for", "agent", required=True, help="Agent to install for")
@click.option("--git-hooks", is_flag=True, default=False, help="Also install git post-checkout/post-merge hooks")
def install(agent, git_hooks):
    """Wire ccindex hook into an AI agent."""
    if agent not in ADAPTERS:
        click.echo(f"Unknown agent: {agent}. Supported: {', '.join(ADAPTERS)}", err=True)
        sys.exit(1)

    root = _find_project_root()
    adapter = ADAPTERS[agent](project_root=root)
    adapter.install()
    click.echo(f"ccindex hook installed for {agent}.")

    if git_hooks:
        from ccindex.git import install_post_checkout_hook, install_post_merge_hook, find_repo_root
        repo = find_repo_root(root)
        if repo:
            install_post_checkout_hook(repo)
            install_post_merge_hook(repo)
            click.echo("Git post-checkout and post-merge hooks installed.")
        else:
            click.echo("Warning: not a git repo — git hooks not installed.")


@main.command()
@click.option("--for", "agent", required=True, help="Agent to uninstall from")
def uninstall(agent, ):
    """Remove ccindex hook from an AI agent."""
    if agent not in ADAPTERS:
        click.echo(f"Unknown agent: {agent}. Supported: {', '.join(ADAPTERS)}", err=True)
        sys.exit(1)

    root = _find_project_root()
    adapter = ADAPTERS[agent](project_root=root)
    adapter.uninstall()
    click.echo(f"ccindex hook removed from {agent}.")


@main.command()
def doctor():
    """Verify ccindex setup: model, index, hooks, sqlite-vec."""
    import sys
    root = _find_project_root()
    all_ok = True

    click.echo(f"Python version : {sys.version.split()[0]}")

    # sqlite-vec
    try:
        import sqlite3, sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        click.echo("sqlite-vec     : OK")
    except Exception as e:
        click.echo(f"sqlite-vec     : FAIL ({e})")
        all_ok = False

    # Models
    from ccindex.models import get_model_dir, ModelNotFoundError
    for model_name in ("jina-code-onnx", "reranker-onnx"):
        try:
            d = get_model_dir(model_name)
            click.echo(f"{model_name:20} : OK ({d})")
        except ModelNotFoundError:
            click.echo(f"{model_name:20} : MISSING — run: ccindex update")
            all_ok = False

    # Index
    db_path = _get_db_path(root)
    if db_path.exists():
        from ccindex.index import Index
        idx = Index(db_path)
        state = idx.get_meta("index_state")
        if state == "complete":
            click.echo("Index          : OK")
        else:
            click.echo(f"Index          : WARNING (state={state}) — run: ccindex index")
    else:
        click.echo("Index          : NOT FOUND — run: ccindex index")

    # Hooks
    for agent_name, adapter_cls in ADAPTERS.items():
        adapter = adapter_cls(project_root=root)
        status = "installed" if adapter.is_installed() else "not installed"
        click.echo(f"Hook ({agent_name:15}): {status}")

    sys.exit(0 if all_ok else 1)


@main.command()
def update():
    """Download latest models from GitHub releases."""
    import urllib.request
    import hashlib

    RELEASE_BASE = "https://github.com/dillibk777/ccindex/releases/latest/download"
    MODELS = {
        "jina-code-onnx": ["model.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "reranker-onnx": ["model.onnx", "tokenizer.json"],
    }
    dest_root = Path.home() / ".ccindex" / "models"

    for model_name, files in MODELS.items():
        model_dir = dest_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            url = f"{RELEASE_BASE}/{model_name}/{filename}"
            dest = model_dir / filename
            click.echo(f"Downloading {model_name}/{filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                click.echo(f"Failed: {e}", err=True)

    click.echo("Models updated. Run: ccindex index")


@main.group()
def daemon():
    """Manage the background file watcher daemon."""
    pass


@daemon.command("start")
def daemon_start():
    """Register and start the background file watcher."""
    from ccindex.daemon import register_daemon
    register_daemon()
    click.echo("Daemon registered and started.")


@daemon.command("stop")
def daemon_stop():
    """Stop the background file watcher."""
    from ccindex.daemon import unregister_daemon
    unregister_daemon()
    click.echo("Daemon stopped.")


@daemon.command("status")
def daemon_status():
    """Show daemon status."""
    from ccindex.daemon import get_daemon_status
    status = get_daemon_status()
    click.echo(f"Daemon: {status}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Verify the CLI entry point works**

```bash
ccindex --version
ccindex --help
```

Expected: version string and help text printed.

- [ ] **Step 6: Commit**

```bash
git add src/ccindex/cli.py tests/test_cli.py
git commit -m "feat: full CLI with index, query, status, clear, install, doctor, update, daemon"
```

---

## Task 12: Daemon

**Files:**
- Create: `src/ccindex/daemon.py`

- [ ] **Step 1: Implement `src/ccindex/daemon.py`**

```python
from __future__ import annotations
import sys
import os
import subprocess
from pathlib import Path


def _get_ccindex_exe() -> str:
    return sys.executable.replace("python", "ccindex") if "python" in sys.executable else "ccindex"


def register_daemon() -> None:
    if sys.platform == "darwin":
        _register_launchd()
    elif sys.platform.startswith("linux"):
        _register_systemd()
    elif sys.platform == "win32":
        _register_windows_task()
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def unregister_daemon() -> None:
    if sys.platform == "darwin":
        _unregister_launchd()
    elif sys.platform.startswith("linux"):
        _unregister_systemd()
    elif sys.platform == "win32":
        _unregister_windows_task()


def get_daemon_status() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["launchctl", "list", "com.ccindex.daemon"],
            capture_output=True, text=True
        )
        return "running" if result.returncode == 0 else "stopped"
    elif sys.platform.startswith("linux"):
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "ccindex"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    return "unknown"


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "com.ccindex.daemon.plist"


def _register_launchd():
    exe = _get_ccindex_exe()
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ccindex.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>index</string>
    </array>
    <key>WatchPaths</key>
    <array>
        <string>{Path.cwd()}</string>
    </array>
    <key>ThrottleInterval</key>
    <integer>2</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.ccindex/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.ccindex/daemon.log</string>
</dict>
</plist>"""
    p = _plist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plist)
    subprocess.run(["launchctl", "load", str(p)], check=True)


def _unregister_launchd():
    p = _plist_path()
    if p.exists():
        subprocess.run(["launchctl", "unload", str(p)], check=False)
        p.unlink()


def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "ccindex.service"


def _register_systemd():
    exe = _get_ccindex_exe()
    unit = f"""[Unit]
Description=ccindex file watcher
After=default.target

[Service]
ExecStart={exe} index
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    p = _systemd_unit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "ccindex"], check=True)


def _unregister_systemd():
    subprocess.run(["systemctl", "--user", "disable", "--now", "ccindex"], check=False)
    p = _systemd_unit_path()
    if p.exists():
        p.unlink()


def _register_windows_task():
    exe = _get_ccindex_exe()
    subprocess.run([
        "schtasks", "/create", "/tn", "ccindex_daemon",
        "/tr", f'"{exe}" index',
        "/sc", "onlogon", "/f",
    ], check=True)


def _unregister_windows_task():
    subprocess.run(
        ["schtasks", "/delete", "/tn", "ccindex_daemon", "/f"],
        check=False
    )
```

- [ ] **Step 2: Verify daemon module imports cleanly**

```bash
python -c "from ccindex.daemon import get_daemon_status; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/ccindex/daemon.py
git commit -m "feat: daemon registration for launchd (macOS), systemd (Linux), Task Scheduler (Windows)"
```

---

## Task 13: Model Download Script + Git LFS Setup

**Files:**
- Create: `scripts/download_models.py`

- [ ] **Step 1: Create `scripts/download_models.py`**

```python
"""
Run once during development to download and convert models to ONNX int8.
Requires: pip install optimum[exporters] onnxruntime torch transformers

Output goes to models/ directory — commit via Git LFS.
"""
import subprocess
import hashlib
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export_embedding_model():
    out_dir = MODELS_DIR / "jina-code-onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting jina-embeddings-v2-base-code to ONNX...")
    subprocess.run([
        "optimum-cli", "export", "onnx",
        "--model", "jinaai/jina-embeddings-v2-base-code",
        "--task", "feature-extraction",
        "--optimize", "O2",
        str(out_dir),
    ], check=True)

    print("Quantizing to int8...")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(out_dir), quantization_config=qconfig)

    print(f"Embedding model saved to {out_dir}")
    print(f"  model.onnx SHA256: {sha256(out_dir / 'model.onnx')}")


def export_reranker_model():
    out_dir = MODELS_DIR / "reranker-onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting cross-encoder/ms-marco-MiniLM-L-6-v2 to ONNX...")
    subprocess.run([
        "optimum-cli", "export", "onnx",
        "--model", "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "--task", "text-classification",
        "--optimize", "O2",
        str(out_dir),
    ], check=True)

    print("Quantizing to int8...")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(out_dir), quantization_config=qconfig)

    print(f"Reranker model saved to {out_dir}")
    print(f"  model.onnx SHA256: {sha256(out_dir / 'model.onnx')}")


def write_checksums():
    checksums = {}
    for name in ("jina-code-onnx", "reranker-onnx"):
        model_file = MODELS_DIR / name / "model.onnx"
        if model_file.exists():
            checksums[name] = sha256(model_file)
    out = MODELS_DIR / "checksums.json"
    out.write_text(json.dumps(checksums, indent=2))
    print(f"Checksums written to {out}")


if __name__ == "__main__":
    export_embedding_model()
    export_reranker_model()
    write_checksums()
    print("\nDone. Commit models/ via Git LFS:")
    print("  git lfs track 'models/**/*.onnx'")
    print("  git add models/")
    print("  git commit -m 'chore: add ONNX int8 quantized models via Git LFS'")
```

- [ ] **Step 2: Run the download script (requires HuggingFace access and PyTorch)**

```bash
pip install "optimum[exporters]" torch transformers
python scripts/download_models.py
```

Expected: `models/jina-code-onnx/model.onnx` and `models/reranker-onnx/model.onnx` created.

- [ ] **Step 3: Initialize Git LFS and commit models**

```bash
git lfs install
git lfs track "models/**/*.onnx"
git add .gitattributes models/
git commit -m "chore: add ONNX int8 quantized models via Git LFS"
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Smoke test the full CLI flow**

```bash
cd /tmp && mkdir smoke_test && cd smoke_test
echo 'def authenticate(user, password):\n    return user == "admin"' > auth.py
echo 'def get_user_by_id(user_id):\n    return db.query(user_id)' > users.py
ccindex index
ccindex status
ccindex query "how does authentication work"
```

Expected: `ccindex query` returns `auth.py` with `authenticate` as the top result.

- [ ] **Step 6: Final commit**

```bash
git add scripts/download_models.py
git commit -m "chore: model download and ONNX conversion script for development setup"
```

---

## Full Test Suite

After all tasks complete:

```bash
pytest tests/ -v --cov=ccindex --cov-report=term-missing
```

Expected: All tests pass, coverage >80%.

---

## Sources

- [graphify GitHub](https://github.com/safishamsi/graphify)
- [codegraph GitHub](https://github.com/colbymchenry/codegraph)
- [Graphify + Claude Code integration](https://graphify.net/graphify-claude-code-integration.html)
