import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def tmp_project(tmp_path):
    """A temporary directory acting as a project root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    return tmp_path


@pytest.fixture
def tmp_git_project(tmp_project):
    """Temporary project with a git repo initialized."""
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_project, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_project, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_project, check=True, capture_output=True)
    return tmp_project
