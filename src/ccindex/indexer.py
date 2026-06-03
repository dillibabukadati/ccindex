from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable
import numpy as np
from ccindex.config import Config
from ccindex.walker import walk_project
from ccindex.chunker import chunk_file, Chunk
from ccindex.index import Index
from ccindex.models import EmbeddingModel

_t_start = time.time()

def _dbg(msg: str):
    print(f"[indexer {time.time()-_t_start:6.2f}s] {msg}", file=sys.stderr, flush=True)


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

    def _embed_and_write(self, chunks: list[Chunk], done_so_far: int, total: int):
        batch_size = self.config.batch_size
        sorted_chunks = sorted(chunks, key=lambda c: len(c.chunk_text))
        n_batches = (len(sorted_chunks) + batch_size - 1) // batch_size
        _dbg(f"_embed_and_write: {len(chunks)} chunks, {n_batches} batches of {batch_size}")
        for i in range(0, len(sorted_chunks), batch_size):
            batch = sorted_chunks[i:i + batch_size]
            _t_batch = time.time()
            embeddings = self.model.embed([c.chunk_text for c in batch])
            t_embed = time.time() - _t_batch
            self._index.upsert_chunks(list(zip(batch, embeddings)))
            t_write = time.time() - _t_batch - t_embed
            batch_num = i // batch_size + 1
            _dbg(f"  batch {batch_num}/{n_batches}: embed={t_embed:.2f}s write={t_write:.2f}s")
            if self.progress_cb and total:
                self.progress_cb(min(done_so_far + i + batch_size, total), total)

    def run_full(self, show_progress: bool = False):
        _dbg(f"run_full: root={self.root}")
        self._ensure_gitignore()
        self._index.set_meta("index_state", "partial")

        files = list(walk_project(self.root, self.config))
        _dbg(f"walk done: {len(files)} files")

        all_chunks: list[Chunk] = []
        for path in files:
            all_chunks.extend(chunk_file(path, self.root, self.config))
        _dbg(f"chunk done: {len(all_chunks)} chunks")

        total = len(all_chunks)
        if self.progress_cb:
            self.progress_cb(0, total)

        self._embed_and_write(all_chunks, 0, total)

        self._index.set_meta("index_state", "complete")
        _dbg("run_full complete")

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
