from __future__ import annotations
from pathlib import Path
from typing import Iterator
import json
import pathspec
from ccindex.config import Config

_ALWAYS_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "dist", "build",
    ".ccindex", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "coverage", ".coverage",
    "models",  # bundled ONNX model files — not project source
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
        data = json.loads(tsconfig.read_text())
        return data.get("compilerOptions", {}).get("outDir")
    except (json.JSONDecodeError, OSError):
        return None


def walk_project(root: Path, config: Config) -> Iterator[Path]:
    gitignore = _load_spec(root, ".gitignore")
    ccignore = _load_spec(root, ".ccindexignore")
    ts_out_dir = _get_tsconfig_out_dir(root)
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
