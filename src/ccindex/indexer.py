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
        self._index = Index(self._db_path, embedding_dim=model.dim)

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

    def _embed_and_write(self, chunks: list[Chunk], done_so_far: int, total: int):
        """Embed chunks in length-sorted batches and write to DB immediately.

        Sorting by text length before batching keeps each batch's padded seq_len
        near the actual chunk length, avoiding O(seq²) wasted attention on padding.
        e.g. batch of 32 short chunks pads to ~160 tokens (fast) instead of 512 (slow).
        """
        batch_size = self.config.batch_size
        # Sort by length so each batch pads to its own max, not the global max
        sorted_chunks = sorted(chunks, key=lambda c: len(c.chunk_text))
        for i in range(0, len(sorted_chunks), batch_size):
            batch = sorted_chunks[i:i + batch_size]
            embeddings = self.model.embed([c.chunk_text for c in batch])
            self._index.upsert_chunks(list(zip(batch, embeddings)))
            if self.progress_cb and total:
                self.progress_cb(min(done_so_far + i + batch_size, total), total)

    def run_full(self, show_progress: bool = False):
        self._ensure_gitignore()
        self._index.set_meta("index_state", "partial")

        files = list(walk_project(self.root, self.config))

        # Collect all chunks first (lightweight — just text, no embeddings yet)
        all_chunks: list[Chunk] = []
        for path in files:
            all_chunks.extend(chunk_file(path, self.root, self.config))

        total = len(all_chunks)
        if self.progress_cb:
            self.progress_cb(0, total)

        # Embed globally in batches and write to DB as we go (avoids holding all embeddings in RAM)
        self._embed_and_write(all_chunks, 0, total)

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

        all_chunks: list[Chunk] = []
        for path in to_update:
            chunks = chunk_file(path, self.root, self.config)
            if not chunks:
                self._index.delete_by_path(path.relative_to(self.root).as_posix())
            else:
                all_chunks.extend(chunks)

        if all_chunks:
            self._embed_and_write(all_chunks, 0, 0)
