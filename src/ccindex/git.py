from __future__ import annotations
import subprocess
from pathlib import Path


def _run(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def find_repo_root(start: Path) -> Path | None:
    out = _run(["git", "rev-parse", "--show-toplevel"], start)
    return Path(out) if out else None


def get_current_commit(repo_root: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], repo_root)


def get_current_branch(repo_root: Path) -> str | None:
    return _run(["git", "branch", "--show-current"], repo_root)


def get_changed_files(repo_root: Path, from_commit: str, to_commit: str) -> list[str]:
    if from_commit == to_commit:
        return []
    out = _run(
        ["git", "diff", "--name-only", from_commit, to_commit],
        repo_root,
    )
    if not out:
        return []
    return [f for f in out.splitlines() if f]


def is_merge_in_progress(repo_root: Path) -> bool:
    return (repo_root / ".git" / "MERGE_HEAD").exists()


def install_post_checkout_hook(repo_root: Path) -> bool:
    hook_path = repo_root / ".git" / "hooks" / "post-checkout"
    script = "#!/bin/sh\nccindex index\n"
    try:
        hook_path.write_text(script)
        hook_path.chmod(0o755)
        return True
    except OSError:
        return False


def install_post_merge_hook(repo_root: Path) -> bool:
    hook_path = repo_root / ".git" / "hooks" / "post-merge"
    script = "#!/bin/sh\nccindex index\n"
    try:
        hook_path.write_text(script)
        hook_path.chmod(0o755)
        return True
    except OSError:
        return False
