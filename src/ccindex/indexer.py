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
