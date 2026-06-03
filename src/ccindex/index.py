from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3  # type: ignore[no-redef]

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
    def __init__(self, db_path: Path, embedding_dim: int = 768):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim
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
            self._conn.executescript("""
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunks_vec;
            """)
            self._create_schema()
            self.set_meta("schema_version", self._SCHEMA_VERSION)
            self.set_meta("index_state", "partial")

    def _create_schema(self):
        self._conn.executescript(f"""
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
                embedding FLOAT[{self._embedding_dim}]
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
               WHERE v.embedding MATCH ? AND v.k = ?
               ORDER BY v.distance""",
            (_serialize(embedding), top_k),
        ).fetchall()
        return [SearchResult(
            chunk_id=r[0], file_path=r[1], start_line=r[2], end_line=r[3],
            symbol=r[4], lang=r[5], chunk_text=r[6],
            score=1.0 - float(r[7]),
        ) for r in rows]

    def fts_search(self, query: str, top_k: int) -> list[SearchResult]:
        safe_query = query.replace('"', '""')
        try:
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
        except sqlite3.OperationalError:
            return []
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
