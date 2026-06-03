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
