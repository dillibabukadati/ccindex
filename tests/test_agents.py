import pytest
import json
from pathlib import Path
from unittest.mock import patch
from ccindex.agents.claude_code import ClaudeCodeAdapter
from ccindex.agents.gemini_cli import GeminiCliAdapter


@pytest.fixture
def mock_user_home(tmp_path):
    """Redirect ~/.claude writes to tmp_path to avoid touching real user settings."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    with patch.object(Path, "home", return_value=fake_home):
        yield fake_home


def test_claude_code_install_writes_hook(mock_user_home, tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    settings_path = mock_user_home / ".claude" / "settings.json"
    assert settings_path.exists()
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    assert any("ccindex" in h["command"] for h in hooks)


def test_claude_code_install_writes_slash_command(mock_user_home, tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    command_path = mock_user_home / ".claude" / "commands" / "ccindex.md"
    assert command_path.exists()
    assert "ccindex query" in command_path.read_text()


def test_claude_code_install_idempotent(mock_user_home, tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    adapter.install()
    settings_path = mock_user_home / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    ccindex_hooks = [h for h in hooks if "ccindex" in h["command"]]
    assert len(ccindex_hooks) == 1


def test_claude_code_install_preserves_existing_hooks(mock_user_home, tmp_path):
    settings_path = mock_user_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    existing = {
        "hooks": {
            "UserPromptSubmit": [{
                "matcher": ".*",
                "hooks": [{"type": "command", "command": "other-tool"}]
            }]
        }
    }
    settings_path.write_text(json.dumps(existing))
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    commands = [h["command"] for h in hooks]
    assert "other-tool" in commands
    assert any("ccindex" in c for c in commands)


def test_claude_code_uninstall_removes_hook_and_command(mock_user_home, tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    adapter.uninstall()
    settings_path = mock_user_home / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    hooks = data.get("hooks", {}).get("UserPromptSubmit", [{}])[0].get("hooks", [])
    assert not any("ccindex" in h.get("command", "") for h in hooks)
    command_path = mock_user_home / ".claude" / "commands" / "ccindex.md"
    assert not command_path.exists()


def test_claude_code_check_returns_true_after_install(mock_user_home, tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    assert not adapter.is_installed()
    adapter.install()
    assert adapter.is_installed()
