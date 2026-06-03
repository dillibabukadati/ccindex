from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
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


def _get_parser_and_language(lang: str):
    """Return (Parser, Language) for the given lang, or (None, None) on failure."""
    try:
        from tree_sitter import Language, Parser
        if lang == "python":
            import tree_sitter_python as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "javascript":
            import tree_sitter_javascript as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "typescript":
            import tree_sitter_typescript as ts_mod
            ts_lang = Language(ts_mod.language_typescript())
        elif lang == "go":
            import tree_sitter_go as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "rust":
            import tree_sitter_rust as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "java":
            import tree_sitter_java as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "c":
            import tree_sitter_c as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "cpp":
            import tree_sitter_cpp as ts_mod
            ts_lang = Language(ts_mod.language())
        elif lang == "ruby":
            import tree_sitter_ruby as ts_mod
            ts_lang = Language(ts_mod.language())
        else:
            return None, None
        return Parser(ts_lang), ts_lang
    except (ImportError, Exception):
        return None, None


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


def _treesitter_chunks(
    path: Path, rel: str, lang: str, source: str, mtime: float
) -> list[Chunk]:
    parser, ts_lang = _get_parser_and_language(lang)
    if parser is None or ts_lang is None:
        return []

    query_str = _SYMBOL_QUERIES.get(lang, "")
    if not query_str:
        return []

    try:
        from tree_sitter import Query, QueryCursor
        tree = parser.parse(bytes(source, "utf-8"))
        lines = source.splitlines()
        query = Query(ts_lang, query_str)
        cursor = QueryCursor(query)
        # matches returns list of (pattern_id, {capture_name: [nodes]})
        matches = cursor.matches(tree.root_node)
    except Exception:
        return []

    chunks: list[Chunk] = []

    for _pattern_id, capture_dict in matches:
        node_list = capture_dict.get("node", [])
        name_list = capture_dict.get("name", [])

        for node in node_list:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            node_lines = lines[node.start_point[0]:node.end_point[0] + 1]

            # Find the matching name node (should be within node bounds)
            symbol = None
            for name_node in name_list:
                if node.start_point[0] <= name_node.start_point[0] <= node.end_point[0]:
                    symbol = source[name_node.start_byte:name_node.end_byte]
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
    window_tokens: int, overlap_tokens: int,
) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    window_chars = window_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    # Build cumulative char start positions for each line
    line_starts: list[int] = []
    char = 0
    for line in lines:
        line_starts.append(char)
        char += len(line)

    text = source
    chunks: list[Chunk] = []
    pos = 0

    while pos < len(text):
        end = min(pos + window_chars, len(text))
        slice_text = text[pos:end]

        # Determine start/end line numbers (1-based)
        start_line = 1
        for i, ls in enumerate(line_starts):
            if ls <= pos:
                start_line = i + 1
            else:
                break

        end_line = len(lines)
        for i, ls in enumerate(line_starts):
            if ls >= end:
                end_line = i + 1
                break

        prefix = ""
        if lang == "markdown" and pos > 0:
            heading_lines = [ln.strip() for ln in slice_text.splitlines() if ln.startswith("#")]
            if not heading_lines:
                prev_headings = [ln.strip() for ln in text[:pos].splitlines() if ln.startswith("#")]
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

        if end >= len(text):
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
        # Fallback to sliding window if tree-sitter finds nothing or unavailable
        return _sliding_window_chunks(path, rel, lang, source, mtime, 128, 32)

    if lang in _DOC_LANGS:
        return _sliding_window_chunks(
            path, rel, lang, source, mtime,
            _SLIDING_WINDOW_TOKENS, _SLIDING_OVERLAP_TOKENS,
        )

    if lang in _CONFIG_LANGS:
        if len(source.encode()) <= _CONFIG_MAX_BYTES:
            return [Chunk(
                file_path=rel,
                start_line=1,
                end_line=source.count("\n") + 1,
                symbol=None,
                lang=lang,
                chunk_text=source,
                file_mtime=mtime,
            )]
        return _sliding_window_chunks(
            path, rel, lang, source, mtime,
            _CONFIG_WINDOW_TOKENS, _CONFIG_OVERLAP_TOKENS,
        )

    # Unknown extension — sliding window fallback
    return _sliding_window_chunks(path, rel, lang, source, mtime, 128, 32)
