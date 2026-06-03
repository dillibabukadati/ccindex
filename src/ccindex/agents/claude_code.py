from __future__ import annotations
import json
from pathlib import Path
from ccindex.agents.base import AgentAdapter

_HOOK_COMMAND = "ccindex query --top 5 --format hook"
_TIMEOUT_MS = 3000
_COMMAND_NAME = "ccindex"
_COMMAND_CONTENT = """\
Search the local ccindex for code relevant to: $ARGUMENTS

Run this shell command and show the results to the user:
  ccindex query "$ARGUMENTS" --format text

If the output says "No index found", tell the user to run `ccindex index` in their project first.
If the output says "No results found", let the user know no relevant code was found for that query.
"""


class ClaudeCodeAdapter(AgentAdapter):
    @property
    def _settings_path(self) -> Path:
        return Path.home() / ".claude" / "settings.json"

    @property
    def _command_path(self) -> Path:
        return Path.home() / ".claude" / "commands" / f"{_COMMAND_NAME}.md"

    def _load_settings(self) -> dict:
        if self._settings_path.exists():
            try:
                return json.loads(self._settings_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_settings(self, data: dict):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(data, indent=2))

    def install(self):
        # 1. UserPromptSubmit hook → ~/.claude/settings.json
        data = self._load_settings()
        data.setdefault("hooks", {})
        data["hooks"].setdefault("UserPromptSubmit", [{"matcher": ".*", "hooks": []}])

        entry = data["hooks"]["UserPromptSubmit"][0]
        entry.setdefault("hooks", [])

        if not any(_HOOK_COMMAND in h.get("command", "") for h in entry["hooks"]):
            entry["hooks"].append({
                "type": "command",
                "command": _HOOK_COMMAND,
                "timeout": _TIMEOUT_MS,
            })

        self._save_settings(data)

        # 2. /ccindex slash command → ~/.claude/commands/ccindex.md
        self._command_path.parent.mkdir(parents=True, exist_ok=True)
        self._command_path.write_text(_COMMAND_CONTENT)

    def uninstall(self):
        # Remove hook
        data = self._load_settings()
        try:
            hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
            data["hooks"]["UserPromptSubmit"][0]["hooks"] = [
                h for h in hooks if _HOOK_COMMAND not in h.get("command", "")
            ]
            self._save_settings(data)
        except (KeyError, IndexError):
            pass

        # Remove slash command
        if self._command_path.exists():
            self._command_path.unlink()

    def is_installed(self) -> bool:
        data = self._load_settings()
        try:
            hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
            hook_ok = any(_HOOK_COMMAND in h.get("command", "") for h in hooks)
        except (KeyError, IndexError):
            hook_ok = False
        return hook_ok and self._command_path.exists()
