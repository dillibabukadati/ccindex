from __future__ import annotations
from pathlib import Path
from typing import Iterator
import json
import os
import pathspec
from ccindex.config import Config

_ALWAYS_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "dist", "build",
    ".ccindex", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "coverage", ".coverage",
    "models",  # bundled ONNX model files — not project source
    # Mobile / native dependency dirs
    "Pods", "DerivedData",  # CocoaPods + Xcode build artifacts
    ".gradle", "gradle",    # Android build system
    # Cache / generated dirs
    ".expo", ".expo-shared", ".next", ".nuxt", ".output",
    ".turbo", ".parcel-cache", ".cache", "tmp", ".tmp",
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

_BINARY_EXTENSIONS = frozenset({
    ".ipa", ".apk", ".aab", ".dex",  # mobile archives/binaries
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",  # archives
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",  # images
    ".ttf", ".otf", ".woff", ".woff2",  # fonts
    ".mp4", ".mov", ".mp3", ".wav",  # media
    ".pdf", ".dmg", ".pkg",  # documents/installers
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
    ts_out_dir_clean = ts_out_dir.lstrip("./") if ts_out_dir else None
    extra_spec = (
        pathspec.PathSpec.from_lines("gitwildmatch", config.ignore_patterns)
        if config.ignore_patterns else None
    )
    max_bytes = config.max_file_size_kb * 1024

    # os.walk with followlinks=False prevents infinite loops from circular symlinks
    # and stops the walker from escaping the project root via symlinks
    for dirpath_str, dirnames, filenames in os.walk(root, followlinks=False):
        dirpath = Path(dirpath_str)
        try:
            rel_dir = dirpath.relative_to(root)
        except ValueError:
            continue

        # Prune dirs in-place so os.walk never descends into them
        dirnames[:] = [
            d for d in dirnames
            if d not in _ALWAYS_SKIP_DIRS
            and (ts_out_dir_clean is None or not (rel_dir == Path(".") and d == ts_out_dir_clean))
        ]

        for filename in filenames:
            path = dirpath / filename
            rel = rel_dir / filename
            rel_str = rel.as_posix()

            # Skip gitignored paths
            if gitignore and gitignore.match_file(rel_str):
                continue
            if ccignore and ccignore.match_file(rel_str):
                continue
            if extra_spec and extra_spec.match_file(rel_str):
                continue

            # Skip lock / secret files
            if filename in _LOCK_FILES:
                continue
            if filename.startswith(".env") or any(filename.endswith(s) for s in _SECRET_PATTERNS):
                continue

            # Skip minified/generated
            name_lower = filename.lower()
            if any(name_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
                continue
            if any(filename.endswith(s) for s in _SKIP_SUFFIXES):
                continue

            # Fast extension filter — no file I/O needed
            suffix_lower = Path(filename).suffix.lower()
            if suffix_lower in _BINARY_EXTENSIONS:
                continue
            if suffix_lower not in _TEXT_EXTENSIONS and suffix_lower != "":
                continue

            # Skip oversized files
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue

            # Skip binary files (content check)
            if _is_binary(path):
                continue

            yield path
