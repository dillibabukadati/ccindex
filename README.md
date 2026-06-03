# ccindex

**Cursor-like semantic code search for AI coding agents — runs entirely on your machine.**

ccindex indexes your codebase using a local ONNX code embedding model and injects the most relevant code chunks into every AI message automatically. No API calls. No MCP server. No cloud. Just fast, accurate context — the way Cursor does it, but for your terminal-based agents.

```
$ ccindex index          # index your project once
$ ccindex install --for claude-code   # wire it into Claude Code globally
# that's it — every message now gets relevant code context prepended automatically
```

---

## How it works

```
Your message
     │
     ▼
┌─────────────────────────────────────────────────┐
│  UserPromptSubmit hook (fires before every msg) │
│                                                 │
│  1. Embed query  →  jina-embeddings-v2-code     │
│  2. ANN search   →  sqlite-vec (top 50)         │
│  3. Keyword search → FTS5 (top 20)              │
│  4. Deduplicate                                 │
│  5. Rerank       →  ms-marco-MiniLM cross-enc   │
│  6. Token-cap at 1500 tokens                    │
│  7. Inject as [ccindex context] block           │
└─────────────────────────────────────────────────┘
     │
     ▼
Claude Code (sees your message + relevant code)
```

**Two-stage retrieval — same as Cursor:**
- Stage 1: Approximate nearest-neighbor search over 768-dim code embeddings (fast, high recall)
- Stage 2: Cross-encoder reranking to pick the truly relevant chunks (precise)

**Branch-aware indexing:**
- Stores the current git commit hash in the index
- On branch switch, diffs against the stored hash and re-embeds only changed files
- Query in main, switch to feature branch, query again — always gets the right context

---

## Features

- **Fully offline** — ONNX models bundled, no HuggingFace at runtime
- **Zero-token overhead** — injects context via hook, not MCP (MCP adds tool-call tokens on every message)
- **Incremental re-index** — only re-embeds changed files; lazy re-index on every query
- **Tree-sitter chunking** — extracts functions, methods, classes as individual chunks for 13 languages
- **Sliding window fallback** — Markdown, YAML, TOML, plain text get 40-line overlapping windows
- **Jupyter notebook support** — each cell is a chunk
- **FTS5 keyword search** — hybrid retrieval catches exact identifiers the embedding model might miss
- **Git hook integration** — optional post-checkout/post-merge hooks for instant re-index on branch switch
- **Monorepo support** — walks up from CWD to find the nearest `.ccindex/` directory
- **Per-project config** — `.ccindex/config.toml` overrides for `top_k`, `token_cap`, ignore patterns

---

## Installation

```bash
pip install ccindex
# or
uv add ccindex
```

Models are bundled with the package (~240MB total). If you installed via pip without Git LFS:

```bash
ccindex update   # downloads models from GitHub releases
```

Verify everything is working:

```bash
ccindex doctor
```

---

## Quick start

```bash
# 1. Go to your project
cd ~/my-project

# 2. Index it
ccindex index

# 3. Wire into Claude Code (one-time, global)
ccindex install --for claude-code

# 4. Start Claude Code — context flows automatically
claude
```

From this point on, every message you send in Claude Code gets relevant code chunks prepended. No `/ccindex` invocation needed — it just works.

---

## Agent integrations

### Claude Code (recommended)

```bash
ccindex install --for claude-code
```

This does two things:
1. Adds a `UserPromptSubmit` hook to `~/.claude/settings.json` — fires on every message, injects context automatically
2. Creates `~/.claude/commands/ccindex.md` — registers `/ccindex <query>` as a slash command for explicit targeted search

The hook is user-level (not project-level), so you install once and every project that has been indexed gets context automatically. Projects without an index are silently skipped.

### Gemini CLI

```bash
ccindex install --for gemini-cli
```

### Antigravity

```bash
ccindex install --for antigravity
```

---

## Commands

| Command | Description |
|---|---|
| `ccindex index` | Index or re-index the current project (incremental) |
| `ccindex query "text"` | Search the index manually |
| `ccindex status` | Show files indexed, branch, index size |
| `ccindex doctor` | Verify models, sqlite-vec, hooks |
| `ccindex install --for <agent>` | Wire hook + slash command into an agent |
| `ccindex uninstall --for <agent>` | Remove hook and slash command |
| `ccindex install --git-hooks` | Also install post-checkout/post-merge git hooks |
| `ccindex clear` | Wipe the index (force full rebuild next time) |
| `ccindex update` | Download latest models from GitHub releases |
| `ccindex daemon start` | Start background file watcher (launchd/systemd) |

### Query options

```bash
ccindex query "auth middleware" --top 10         # return more results
ccindex query "retry logic" --format json        # structured output
ccindex query "how does caching work" --format hook   # hook injection format
```

---

## Configuration

`.ccindex/config.toml` in your project root (or `~/.ccindex/config.toml` globally):

```toml
[query]
top_k = 5              # chunks returned per query
token_cap = 1500       # max tokens injected per message
relevance_threshold = 0.0  # minimum reranker score (0.0 = all results)

[index]
max_file_size_kb = 1024    # skip files larger than this
batch_size = 32            # embedding batch size

[ignore]
patterns = [
    "migrations/",
    "*.generated.ts",
    "fixtures/",
]
```

You can also create a `.ccindexignore` file (same syntax as `.gitignore`) in your project root.

---

## Models

| Model | Size | Purpose |
|---|---|---|
| `jinaai/jina-embeddings-v2-base-code` (quantized) | 154 MB | Code embeddings — 768 dimensions, trained on code |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 86 MB | Reranking — filters ANN candidates by true relevance |

Both models run locally via ONNX Runtime (CPU). No GPU required. A full query (embed + ANN + rerank) takes ~200ms on an M-series Mac.

---

## What gets indexed

Everything that looks like source code or documentation:

- **Code**: Python, JS, TS, TSX, JSX, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, Shell
- **Docs**: Markdown, RST, plain text
- **Config**: JSON, YAML, TOML, HCL, Terraform
- **Notebooks**: `.ipynb` (each cell as a chunk)

**Automatically skipped:**
- `node_modules/`, `.venv/`, `dist/`, `build/`, `__pycache__/`
- `models/` (bundled ONNX artifacts)
- Lock files: `package-lock.json`, `yarn.lock`, `uv.lock`, etc.
- Minified files: `*.min.js`, `*.min.css`
- Generated files: `*_pb2.py`, `*.pb.go`, `*.generated.ts`
- Secret files: `.env`, `*.pem`, `*.key`
- Binary files (detected by content)
- Files over 1MB
- `.gitignore`'d paths

---

## Index storage

The index lives at `<project-root>/.ccindex/index.db` — per-project, isolated, automatically added to `.gitignore`. It is never shared between projects.

Models are stored at `<ccindex-package>/models/` (bundled) or `~/.ccindex/models/` (downloaded via `ccindex update`).

---

## vs. alternatives

| | ccindex | MCP code tools | Cursor | GitHub Copilot |
|---|---|---|---|---|
| Runs locally | ✅ | ✅ | ✅ | ❌ |
| No API tokens spent on retrieval | ✅ | ❌ (tool calls) | ✅ | N/A |
| Works in terminal agents | ✅ | ✅ | ❌ | ❌ |
| Two-stage reranking | ✅ | ❌ | ✅ | ❌ |
| Branch-aware indexing | ✅ | ❌ | ✅ | N/A |
| Hybrid vector + keyword search | ✅ | varies | ✅ | N/A |
| Zero setup for end user | ✅ | ❌ | ✅ | ✅ |

The key difference from MCP-based tools: MCP requires tool call round-trips that consume tokens on every query. ccindex uses a `UserPromptSubmit` hook — context is prepended silently before the message reaches the model, spending zero extra tokens on retrieval.

---

## Development

```bash
git clone https://github.com/dillibk777/ccindex
cd ccindex
uv sync
pytest
```

```bash
# Index this repo itself and test
ccindex index
ccindex query "how does the reranker work"
```

---

## License

MIT
