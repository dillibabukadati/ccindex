from __future__ import annotations
import sys
import json
import os
from pathlib import Path
import click
from ccindex import __version__
from ccindex.config import load_config
from ccindex.agents import ADAPTERS


def _find_project_root() -> Path:
    """Walk up from CWD to find nearest .ccindex/ dir (monorepo support)."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".ccindex").exists():
            return parent
    return current  # fallback: treat CWD as root


def _get_db_path(root: Path) -> Path:
    return root / ".ccindex" / "index.db"


@click.group()
@click.version_option(__version__, prog_name="ccindex")
def main():
    """Local offline-first code indexer for AI coding agents."""
    pass


@main.command()
@click.option("--show-progress", is_flag=True, default=True)
def index(show_progress):
    """Index the current project (incremental if already indexed)."""
    from ccindex.models import get_model_dir, EmbeddingModel, ModelNotFoundError
    from ccindex.indexer import Indexer

    root = _find_project_root()
    config = load_config(root)
    db_path = _get_db_path(root)

    try:
        model_dir = get_model_dir("jina-code-onnx")
    except ModelNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(2)

    model = EmbeddingModel(model_dir)

    if show_progress:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Indexing...", total=None)

            def cb(done, total):
                progress.update(task, completed=done, total=total)

            indexer = Indexer(root=root, config=config, model=model, progress_cb=cb)
            if db_path.exists():
                from ccindex.index import Index
                from ccindex import git
                idx = Index(db_path)
                stored_commit = idx.get_meta("git_commit_hash")

                repo_root = git.find_repo_root(root)
                if repo_root and stored_commit:
                    current_commit = git.get_current_commit(repo_root)
                    if current_commit and current_commit != stored_commit:
                        if git.is_merge_in_progress(repo_root):
                            click.echo("Warning: merge in progress — skipping re-index until resolved.")
                            return
                        changed = git.get_changed_files(repo_root, stored_commit, current_commit)
                        if len(changed) <= config.max_stale_files:
                            indexer.run_incremental(changed_paths=changed)
                            idx.set_meta("git_commit_hash", current_commit)
                            idx.set_meta("git_branch", git.get_current_branch(repo_root) or "")
                        else:
                            click.echo(f"Warning: {len(changed)} files changed — index may be stale. Run `ccindex index` for a full refresh.")
                            return
                    else:
                        indexer.run_incremental()
                else:
                    indexer.run_incremental()
            else:
                click.echo("First run — building index...")
                indexer.run_full(show_progress=True)
                from ccindex import git
                repo_root = git.find_repo_root(root)
                if repo_root:
                    commit = git.get_current_commit(repo_root)
                    branch = git.get_current_branch(repo_root)
                    if commit:
                        indexer.index.set_meta("git_commit_hash", commit)
                    if branch:
                        indexer.index.set_meta("git_branch", branch)

        click.echo("Index updated.")
    else:
        indexer = Indexer(root=root, config=config, model=model)
        if db_path.exists():
            indexer.run_incremental()
        else:
            indexer.run_full()


@main.command()
@click.argument("text", required=False, default=None)
@click.option("--top", default=5, show_default=True)
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json", "hook"]))
def query(text, top, fmt):
    """Search the index for relevant code chunks."""
    from ccindex.models import get_model_dir, EmbeddingModel, Reranker, ModelNotFoundError
    from ccindex.index import Index
    from ccindex.retrieval import Retriever, format_hook_output

    # Resolve query text: argument > env var > stdin
    if text is None:
        text = os.environ.get("CLAUDE_USER_PROMPT", "").strip()
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
    if not text:
        if fmt == "hook":
            sys.exit(1)
        click.echo("Error: Missing argument 'TEXT'.", err=True)
        sys.exit(1)

    root = _find_project_root()
    db_path = _get_db_path(root)
    config = load_config(root)
    config.top_k = top

    if not db_path.exists():
        if fmt == "hook":
            sys.exit(0)  # no index yet — proceed silently, don't block the prompt
        click.echo("No index found. Run: ccindex index", err=True)
        sys.exit(1)

    try:
        embed_dir = get_model_dir("jina-code-onnx")
        rerank_dir = get_model_dir("reranker-onnx")
    except ModelNotFoundError as e:
        if fmt == "hook":
            sys.exit(0)  # models missing — proceed silently
        click.echo(str(e), err=True)
        sys.exit(2)

    _lazy_reindex(root, config, embed_dir, db_path)

    model = EmbeddingModel(embed_dir)
    reranker = Reranker(rerank_dir)
    index = Index(db_path)
    retriever = Retriever(index=index, model=model, reranker=reranker, config=config)

    results = retriever.query(text)

    if fmt == "hook":
        output = format_hook_output(results)
        if output:
            click.echo(output)
        sys.exit(0 if results else 1)
    elif fmt == "json":
        click.echo(json.dumps([{
            "file": r.file_path, "start_line": r.start_line,
            "end_line": r.end_line, "symbol": r.symbol,
            "score": r.score, "text": r.chunk_text,
        } for r in results]))
    else:
        if not results:
            click.echo("No results found.")
            sys.exit(1)
        for r in results:
            loc = f"{r.file_path}"
            if r.start_line:
                loc += f":{r.start_line}"
            if r.symbol:
                loc += f" ({r.symbol})"
            click.echo(f"\n── {loc}")
            click.echo(r.chunk_text[:500])


def _lazy_reindex(root: Path, config, embed_dir: Path, db_path: Path):
    from ccindex.models import EmbeddingModel
    from ccindex.index import Index
    from ccindex.indexer import Indexer
    from ccindex import git

    try:
        idx = Index(db_path)
        stored_commit = idx.get_meta("git_commit_hash")
        repo_root = git.find_repo_root(root)

        if repo_root and stored_commit:
            current_commit = git.get_current_commit(repo_root)
            if current_commit and current_commit != stored_commit:
                if git.is_merge_in_progress(repo_root):
                    return
                changed = git.get_changed_files(repo_root, stored_commit, current_commit)
                if len(changed) <= config.max_stale_files:
                    model = EmbeddingModel(embed_dir)
                    indexer = Indexer(root=root, config=config, model=model)
                    indexer.run_incremental(changed_paths=changed)
                    idx.set_meta("git_commit_hash", current_commit)
                    idx.set_meta("git_branch", git.get_current_branch(repo_root) or "")
    except Exception:
        pass


@main.command()
def status():
    """Show index statistics."""
    root = _find_project_root()
    db_path = _get_db_path(root)

    if not db_path.exists():
        click.echo("No index found. Run: ccindex index")
        return

    from ccindex.index import Index
    import os as _os
    idx = Index(db_path)
    mtimes = idx.get_all_mtimes()
    size_mb = _os.path.getsize(db_path) / (1024 * 1024)
    branch = idx.get_meta("git_branch") or "unknown"
    commit = (idx.get_meta("git_commit_hash") or "")[:8]

    click.echo(f"Files indexed : {len(mtimes)}")
    click.echo(f"Index size    : {size_mb:.1f} MB")
    click.echo(f"Branch        : {branch} @ {commit}")
    click.echo(f"State         : {idx.get_meta('index_state') or 'unknown'}")


@main.command()
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clear(yes):
    """Wipe the index database."""
    root = _find_project_root()
    db_path = _get_db_path(root)

    if not db_path.exists():
        click.echo("No index to clear.")
        return

    if not yes:
        click.confirm("This will delete the index. Continue?", abort=True)

    db_path.unlink()
    click.echo("Index cleared.")


@main.command()
@click.option("--for", "agent", required=True, help="Agent to install for")
@click.option("--git-hooks", is_flag=True, default=False, help="Also install git post-checkout/post-merge hooks")
def install(agent, git_hooks):
    """Wire ccindex hook into an AI agent."""
    if agent not in ADAPTERS:
        click.echo(f"Unknown agent: {agent}. Supported: {', '.join(ADAPTERS)}", err=True)
        sys.exit(1)

    root = _find_project_root()
    adapter = ADAPTERS[agent](project_root=root)
    adapter.install()
    click.echo(f"ccindex hook installed for {agent}.")

    if git_hooks:
        from ccindex.git import install_post_checkout_hook, install_post_merge_hook, find_repo_root
        repo = find_repo_root(root)
        if repo:
            install_post_checkout_hook(repo)
            install_post_merge_hook(repo)
            click.echo("Git post-checkout and post-merge hooks installed.")
        else:
            click.echo("Warning: not a git repo — git hooks not installed.")


@main.command()
@click.option("--for", "agent", required=True, help="Agent to uninstall from")
def uninstall(agent):
    """Remove ccindex hook from an AI agent."""
    if agent not in ADAPTERS:
        click.echo(f"Unknown agent: {agent}. Supported: {', '.join(ADAPTERS)}", err=True)
        sys.exit(1)

    root = _find_project_root()
    adapter = ADAPTERS[agent](project_root=root)
    adapter.uninstall()
    click.echo(f"ccindex hook removed from {agent}.")


@main.command()
def doctor():
    """Verify ccindex setup: model, index, hooks, sqlite-vec."""
    root = _find_project_root()
    all_ok = True

    click.echo(f"Python version : {sys.version.split()[0]}")

    try:
        try:
            import pysqlite3 as sqlite3_mod
        except ImportError:
            import sqlite3 as sqlite3_mod
        import sqlite_vec
        conn = sqlite3_mod.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        click.echo("sqlite-vec     : OK")
    except Exception as e:
        click.echo(f"sqlite-vec     : FAIL ({e})")
        all_ok = False

    from ccindex.models import get_model_dir, ModelNotFoundError
    for model_name in ("jina-code-onnx", "reranker-onnx"):
        try:
            d = get_model_dir(model_name)
            click.echo(f"{model_name:20} : OK ({d})")
        except ModelNotFoundError:
            click.echo(f"{model_name:20} : MISSING — run: ccindex update")
            all_ok = False

    db_path = _get_db_path(root)
    if db_path.exists():
        from ccindex.index import Index
        idx = Index(db_path)
        state = idx.get_meta("index_state")
        if state == "complete":
            click.echo("Index          : OK")
        else:
            click.echo(f"Index          : WARNING (state={state}) — run: ccindex index")
    else:
        click.echo("Index          : NOT FOUND — run: ccindex index")

    for agent_name, adapter_cls in ADAPTERS.items():
        adapter = adapter_cls(project_root=root)
        inst = "installed" if adapter.is_installed() else "not installed"
        click.echo(f"Hook ({agent_name:15}): {inst}")

    sys.exit(0 if all_ok else 1)


@main.command()
def update():
    """Download latest models from GitHub releases."""
    import urllib.request

    RELEASE_BASE = "https://github.com/dillibk777/ccindex/releases/latest/download"
    MODELS = {
        "jina-code-onnx": ["model.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "reranker-onnx": ["model.onnx", "tokenizer.json"],
    }
    dest_root = Path.home() / ".ccindex" / "models"

    for model_name, files in MODELS.items():
        model_dir = dest_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            url = f"{RELEASE_BASE}/{model_name}/{filename}"
            dest = model_dir / filename
            click.echo(f"Downloading {model_name}/{filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                click.echo(f"Failed: {e}", err=True)

    click.echo("Models updated. Run: ccindex index")


@main.group()
def daemon():
    """Manage the background file watcher daemon."""
    pass


@daemon.command("start")
def daemon_start():
    """Register and start the background file watcher."""
    from ccindex.daemon import register_daemon
    register_daemon()
    click.echo("Daemon registered and started.")


@daemon.command("stop")
def daemon_stop():
    """Stop the background file watcher."""
    from ccindex.daemon import unregister_daemon
    unregister_daemon()
    click.echo("Daemon stopped.")


@daemon.command("status")
def daemon_status():
    """Show daemon status."""
    from ccindex.daemon import get_daemon_status
    status = get_daemon_status()
    click.echo(f"Daemon: {status}")
