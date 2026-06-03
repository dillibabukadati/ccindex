# tests/test_agents.py
import pytest
import json
from pathlib import Path
from ccindex.agents.claude_code import ClaudeCodeAdapter
from ccindex.agents.gemini_cli import GeminiCliAdapter


def test_claude_code_install_writes_hook(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    assert settings_path.exists()
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    assert any("ccindex" in h["command"] for h in hooks)


def test_claude_code_install_idempotent(tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    adapter.install()
    settings_path = tmp_path / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
    ccindex_hooks = [h for h in hooks if "ccindex" in h["command"]]
    assert len(ccindex_hooks) == 1


def test_claude_code_install_preserves_existing_hooks(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
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


def test_claude_code_uninstall_removes_hook(tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    adapter.install()
    adapter.uninstall()
    settings_path = tmp_path / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    hooks = data.get("hooks", {}).get("UserPromptSubmit", [{}])[0].get("hooks", [])
    assert not any("ccindex" in h.get("command", "") for h in hooks)


def test_claude_code_check_returns_true_after_install(tmp_path):
    adapter = ClaudeCodeAdapter(project_root=tmp_path)
    assert not adapter.is_installed()
    adapter.install()
    assert adapter.is_installed()
