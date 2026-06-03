# ccindex — Design Spec
**Date:** 2026-06-03  
**Status:** Approved

---

## Overview

`ccindex` is a local, offline-first code indexing and semantic search CLI tool. It indexes any project using local ONNX-quantized embedding models, stores a per-project vector index in SQLite, and injects relevant code context into AI coding agents (Claude Code, Gemini CLI, Antigravity, and others) automatically via their hook systems — replicating Cursor's codebase retrieval behaviour without any cloud calls.

---

## Goals

- Cursor-like automatic context injection on every AI agent query
- Fully offline after first model download — no HuggingFace, no cloud calls at runtime
- Per-project isolated indexes — zero cross-repo contamination
- Agent-agnostic — install adapters for any AI coding CLI
- Fast queries under 200ms including lazy incremental re-index

---

## Architecture

```
ccindex (Python CLI)
│
├── ccindex index              → full index on first run, incremental on subsequent runs
├── ccindex query "..."        → lazy re-index changed files → search → rerank → output
├── ccindex daemon start/stop/status  → optional background file watcher
├── ccindex install --for <agent>     → wire hook into agent config
├── ccindex uninstall --for <agent>   → remove hook
├── ccindex doctor             → verify model, index, hook, sqlite-vec
├── ccindex update             → pull latest models from GitHub releases
├── ccindex status             → show index stats
├── ccindex clear              → wipe index.db
└── ccindex --version
```

---

## Data Layout

```
<ccindex package>/
  models/
    jina-code-onnx/            # embedding model, bundled via Git LFS (~130MB)
      model.onnx
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
    reranker-onnx/             # cross-encoder reranker, bundled via Git LFS (~85MB)
      model.onnx
      tokenizer.json

~/.ccindex/
  models/                      # model cache for pip/uv installs (downloaded from GitHub releases)
    jina-code-onnx/
    reranker-onnx/
  config.toml                  # user-level defaults

<project-root>/
  .ccindex/
    index.db                   # SQLite + sqlite-vec, all embeddings + metadata
    config.toml                # per-project overrides (optional)
  .ccindexignore               # per-project exclusion patterns (optional)
```

`.ccindex/` is automatically added to `.gitignore` on first `ccindex index` run.

---

## Models

| Model | Purpose | Format | Size |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-code` | Code + doc embeddings | ONNX int8 | ~130MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking candidates | ONNX int8 | ~85MB |

**Total bundled size:** ~215MB via Git LFS.

**Model delivery:**
- `git clone` → models come via Git LFS directly, zero downloads
- `pip install ccindex` / `uv tool install ccindex` → on first `ccindex index`, models not found → downloaded from **GitHub releases** (not HuggingFace) → cached at `~/.ccindex/models/`
- SHA256 checksum verified on every load — re-download if mismatch or corruption detected
- Python `>= 3.10` required

---

## What Gets Indexed

**Included:**
- Source code files (all languages with tree-sitter support)
- Markdown and documentation files (`.md`, `.txt`, `.rst`)
- Config files (`.json`, `.yaml`, `.toml`, `.sql`, `.env.example`)
- Jupyter notebooks (`.ipynb`) — code cells only, output cells skipped

**Excluded by default:**
- `.gitignore`'d paths
- `node_modules/`, `__pycache__/`, `.git/`, `dist/`, `build/`, `.ccindex/`
- Virtual envs: `.venv/`, `venv/`, `env/`, `.tox/`
- Lock files: `package-lock.json`, `yarn.lock`, `poetry.lock`, `uv.lock`, `Pipfile.lock`
- Minified/generated files: `*.min.js`, `*.min.css`, `*_pb2.py`, `*.pb.go`
- Secret files: `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa`, `id_ed25519`
- Compiled JS output: detected via `tsconfig.json` `outDir` field
- Binary files (detected by content, not extension)
- Files over 500KB

**User-configurable exclusions:**
- `.ccindexignore` in project root (same syntax as `.gitignore`)
- `~/.ccindex/config.toml` for global patterns

---

## Chunking Strategy

### Code files (tree-sitter)
- Extract functions, methods, classes as individual chunks
- Each chunk carries: `file_path`, `start_line`, `end_line`, `symbol_name`, `lang`, `text`
- Functions over 100 lines are split with 20-line overlap
- Fallback to sliding window (128 tokens, 32 overlap) for unsupported languages
- Pre-bundled tree-sitter grammars for top 20 languages (Python, JS, TS, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, C#, Scala, Shell, SQL, YAML, TOML, Markdown, Dockerfile)

### Markdown / docs
- Sliding window: 512 tokens, 64-token overlap
- Heading hierarchy prepended to each chunk for better embedding signal (e.g. `## Auth > ### JWT: ...`)

### Config files
- Under 2KB → whole file as one chunk
- Over 2KB → sliding window: 256 tokens, 32-token overlap

### Jupyter notebooks
- Extract `source` from `code` type cells only
- Each cell = one chunk, tagged with cell index as `symbol`

---

## Index Schema

```sql
-- SQLite + sqlite-vec, WAL mode enabled at creation

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- stores: model_hash, schema_version, index_state ('complete' | 'partial')

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    file_path   TEXT NOT NULL,    -- relative to project root
    start_line  INTEGER,
    end_line    INTEGER,
    symbol      TEXT,
    lang        TEXT,
    chunk_text  TEXT NOT NULL,
    file_mtime  REAL NOT NULL
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_text, symbol, file_path,
    content='chunks', content_rowid='id'
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding FLOAT[768]          -- jina-code outputs 768-dim vectors
);
-- chunks_vec rowid == chunks.id
```

All inserts use bulk transactions (`BEGIN` / `COMMIT`) — never row-by-row.

---

## Retrieval Pipeline

```
User query text
    │
    ▼
Lazy incremental check
    stat() all tracked files (~10ms for 10k files)
    re-embed changed files in batches of 32 (ONNX)
    skip re-index if changed files > max_stale_files (warn: index may be stale)
    │
    ▼
Stage 1a — Semantic recall
    embed query via jina-code ONNX
    ANN search sqlite-vec → top 50 candidates
    │
Stage 1b — Keyword fallback
    FTS5 search → top 20 candidates
    dedupe + merge → ~60 candidates total
    │
    ▼
Stage 2 — Rerank
    cross-encoder ms-marco-MiniLM ONNX scores all candidates
    sort by score descending
    │
    ▼
Relevance threshold
    drop all candidates with score < 0.3
    if no candidates pass → output nothing
    │
    ▼
Token cap
    character-estimate tokens (~4 chars/token)
    keep top results until 1500-token budget exhausted
    │
    ▼
Output top-k chunks (default 5)
```

---

## CLI Commands

### `ccindex index`
- First run: creates `.ccindex/index.db`, adds `.ccindex/` to `.gitignore`, shows onboarding message, displays rich progress bar with ETA
- Subsequent runs: mtime-based incremental — only re-embeds changed/new files, removes deleted files from index
- Detects partial index state (interrupted previous run) and resumes or rebuilds cleanly
- Detects model hash mismatch (ccindex upgraded) and triggers full re-index automatically

### `ccindex query "<text>" [--top N] [--json]`
- Runs lazy incremental re-index first, then full retrieval pipeline
- Default output: formatted text blocks for human reading
- `--json`: machine-readable `[{file, start_line, end_line, symbol, score, text}]`
- Exit codes: `0` = results found, `1` = no results above threshold, `2` = error

### `ccindex daemon start | stop | status`
- `start`: registers OS-level background watcher
  - macOS: launchd plist at `~/Library/LaunchAgents/com.ccindex.daemon.plist`
  - Linux: systemd user unit at `~/.config/systemd/user/ccindex.service`
  - Windows: Task Scheduler entry
- Daemon watches project files via OS events (FSEvents / inotify / ReadDirectoryChangesW)
- Re-indexes changed files immediately with 2s debounce
- Auto-starts on login once registered

### `ccindex install --for <agent>`
- Supported agents: `claude-code`, `gemini-cli`, `antigravity`
- Idempotent: checks if hook already exists before writing
- Adds `.ccindex/` to `.gitignore` if not already present
- Offers to register git post-checkout hook (optional, prompts user)

### `ccindex install --for claude-code` detail
Writes to `.claude/settings.json`:
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "ccindex query --top 5 --format hook",
            "timeout": 3000
          }
        ]
      }
    ]
  }
}
```
Prompt is passed via **stdin** (not shell argument) to avoid any escaping issues with special characters.

Hook output format:
```
[ccindex context]
── src/auth/jwt.py:12-45 (verify_token)
   def verify_token(token: str) -> dict:
       ...

── src/auth/middleware.py:8-31 (auth_required)
   def auth_required(f):
       ...
[end ccindex context]
```

Empty results → no output, no block injected.

### `ccindex uninstall --for <agent>`
Removes hook entry from agent config. Optionally removes git post-checkout hook.

### `ccindex doctor`
Checks and reports:
- Python version >= 3.10
- sqlite-vec extension loads correctly
- Model files present + checksums valid
- `.ccindex/index.db` exists and is not partial
- Hook wired correctly in agent config
- Daemon running (if registered)

### `ccindex update`
- Fetches latest release manifest from GitHub releases
- Compares model checksums — downloads updated models only if changed
- Runs `ccindex index` automatically after model update (model hash changed = full re-index)

### `ccindex status`
Prints: total files indexed, total chunks, index size on disk, model version, last indexed timestamp, daemon status.

### `ccindex clear`
Wipes `index.db`. Does not remove models. Prompts for confirmation.

### `ccindex --version`
Prints package version.

---

## Git Post-Checkout Hook (Optional)

When user opts in during `ccindex install`, writes two git hooks:

`.git/hooks/post-checkout` — fires after `git checkout`, `git switch`, `git stash pop`:
```bash
#!/bin/sh
ccindex index
```

`.git/hooks/post-merge` — fires after `git pull`, `git merge`:
```bash
#!/bin/sh
ccindex index
```

Both trigger incremental re-index only (mtime-based), not a full rebuild.

---

## Configuration

### `~/.ccindex/config.toml` (user-level)
```toml
[query]
top_k = 5
relevance_threshold = 0.3
token_cap = 1500

[index]
max_file_size_kb = 500
batch_size = 32
max_stale_files = 200    # skip lazy re-index above this, warn stale instead

[ignore]
patterns = ["*.generated.ts", "migrations/"]
```

### `.ccindex/config.toml` (per-project, overrides user-level)
Same structure, any key overrides the user-level value.

---

## Edge Cases & Robustness

| Scenario | Handling |
|---|---|
| Model download interrupted | SHA256 checksum on load; re-download from GitHub releases if mismatch |
| Partial index (indexing interrupted) | `index_state` flag in `meta` table; warn + offer resume or rebuild on next run |
| No index exists when hook fires | Exit cleanly with no output — Claude Code proceeds normally |
| Hook timeout | Hard 3s cap; exit with no output if exceeded |
| Empty/no results above threshold | Inject nothing — no empty block |
| Monorepo / nested projects | Walk up from CWD to find nearest `.ccindex/` — use closest one |
| Relative paths in index | All `file_path` values stored relative to project root — portable on rename/move |
| Stale index (watch not running) | Warn once in `ccindex status` and query output if index >1h old |
| Non-UTF8 encoded files | Skip with warning, continue indexing |
| Circular symlinks | Track visited inodes during walk, break cycles |
| Files >500KB | Skip silently (configurable threshold) |
| `ccindex install` run twice | Check existing hook config before writing — fully idempotent |
| Trivial one-word query | Vector search still runs; relevance threshold handles low-signal results naturally |
| Schema migration on upgrade | `schema_version` in `meta` table; auto-rebuild index if version mismatch |
| Compiled JS (tsconfig outDir) | Parse `tsconfig.json` at index time, exclude `outDir` directory |
| Windows paths | All path handling via `pathlib.Path` — no hardcoded `/` separators |
| SQLite concurrent access | WAL mode enabled — daemon writes and hook reads never block each other |
| `uv tool install` model path | Model path resolution checks package install dir first, then `~/.ccindex/models/` |

---

## Future Scope (Not in v1)

- **Option C full integration**: `PreToolUse` hook on `Read`/`Bash` + `/search` slash command skill for mid-session manual queries
- Per-agent output format adapters for Cursor, Kiro, OpenCode
- Duplicate chunk deduplication (cosine similarity >0.97)
- Same-file chunk collapsing in output
- Reranker fine-tuned on code-specific query/passage pairs
- Web UI for browsing the index (`ccindex ui`)
- Team-shared index (read-only remote SQLite via Litestream)

---

## Dependencies

| Package | Purpose |
|---|---|
| `onnxruntime` | Run ONNX models locally |
| `tokenizers` | Fast HuggingFace tokenizers (Rust-based, no torch) |
| `sqlite-vec` | Vector similarity search in SQLite |
| `tree-sitter` + grammars | AST-based code chunking |
| `rich` | Progress bars, formatted terminal output |
| `watchdog` | Cross-platform filesystem events (daemon mode) |
| `tomllib` / `tomli` | Config file parsing |
| `pathlib` | Cross-platform path handling (stdlib) |
| `click` | CLI framework |
