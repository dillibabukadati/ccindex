from __future__ import annotations
import json
from pathlib import Path
from ccindex.agents.base import AgentAdapter

_HOOK_COMMAND = "ccindex query --top 5 --format hook"
_TIMEOUT_MS = 3000


class ClaudeCodeAdapter(AgentAdapter):
    @property
    def _settings_path(self) -> Path:
        return self.project_root / ".claude" / "settings.json"

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

    def uninstall(self):
        data = self._load_settings()
        try:
            hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
            data["hooks"]["UserPromptSubmit"][0]["hooks"] = [
                h for h in hooks if _HOOK_COMMAND not in h.get("command", "")
            ]
            self._save_settings(data)
        except (KeyError, IndexError):
            pass

    def is_installed(self) -> bool:
        data = self._load_settings()
        try:
            hooks = data["hooks"]["UserPromptSubmit"][0]["hooks"]
            return any(_HOOK_COMMAND in h.get("command", "") for h in hooks)
        except (KeyError, IndexError):
            return False
