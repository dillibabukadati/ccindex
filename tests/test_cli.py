# tests/test_cli.py
import pytest
import json
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from ccindex.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_status_no_index(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "No index found" in result.output


def test_clear_no_index(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["clear", "--yes"])
    assert result.exit_code == 0


def test_install_for_claude_code(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["install", "--for", "claude-code"])
    assert result.exit_code == 0


def test_install_unsupported_agent(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["install", "--for", "unknown-agent"])
    assert result.exit_code != 0
    assert "unknown-agent" in result.output or "Unknown agent" in result.output


def test_query_no_index_exits_cleanly(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["query", "how does auth work"])
    assert result.exit_code == 1


def test_query_format_hook_outputs_empty_when_no_index(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["query", "auth", "--format", "hook"])
    assert result.exit_code == 1
    assert result.output.strip() == ""
