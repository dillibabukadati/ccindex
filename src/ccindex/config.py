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
