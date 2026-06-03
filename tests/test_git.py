# tests/test_git.py
import pytest
import subprocess
from pathlib import Path
from ccindex.git import get_current_commit, get_changed_files, is_merge_in_progress, find_repo_root


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True)


def _commit(repo, msg="init"):
    _git(["config", "user.email", "t@t.com"], repo)
    _git(["config", "user.name", "T"], repo)
    (repo / "file.txt").write_text(msg)
    _git(["add", "."], repo)
    _git(["commit", "-m", msg], repo)


def test_get_current_commit_in_repo(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path)
    commit = get_current_commit(tmp_path)
    assert commit is not None and len(commit) == 40


def test_get_current_commit_outside_repo(tmp_path):
    assert get_current_commit(tmp_path) is None


def test_get_changed_files_between_commits(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path, "first")
    first = get_current_commit(tmp_path)

    (tmp_path / "new_file.py").write_text("x = 1")
    _git(["add", "."], tmp_path)
    _git(["commit", "-m", "second"], tmp_path)

    changed = get_changed_files(tmp_path, first, "HEAD")
    assert "new_file.py" in changed


def test_get_changed_files_returns_empty_for_same_commit(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path)
    commit = get_current_commit(tmp_path)
    changed = get_changed_files(tmp_path, commit, commit)
    assert changed == []


def test_is_merge_in_progress_false_normally(tmp_path):
    _git(["init"], tmp_path)
    _commit(tmp_path)
    assert is_merge_in_progress(tmp_path) is False


def test_find_repo_root_from_subdir(tmp_path):
    _git(["init"], tmp_path)
    subdir = tmp_path / "src" / "deep"
    subdir.mkdir(parents=True)
    root = find_repo_root(subdir)
    assert root == tmp_path


def test_find_repo_root_returns_none_outside_repo(tmp_path):
    assert find_repo_root(tmp_path) is None
