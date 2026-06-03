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
