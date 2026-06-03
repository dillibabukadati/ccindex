from __future__ import annotations
import json
from pathlib import Path
from ccindex.agents.base import AgentAdapter

_HOOK_COMMAND = "ccindex query --top 5 --format hook"


class GeminiCliAdapter(AgentAdapter):
    @property
    def _settings_path(self) -> Path:
        return self.project_root / ".gemini" / "settings.json"

    def install(self):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self._settings_path.exists():
            try:
                data = json.loads(self._settings_path.read_text())
            except json.JSONDecodeError:
                pass
        data.setdefault("hooks", {})
        data["hooks"].setdefault("userPromptSubmit", [])
        if _HOOK_COMMAND not in data["hooks"]["userPromptSubmit"]:
            data["hooks"]["userPromptSubmit"].append(_HOOK_COMMAND)
        self._settings_path.write_text(json.dumps(data, indent=2))

    def uninstall(self):
        if not self._settings_path.exists():
            return
        try:
            data = json.loads(self._settings_path.read_text())
            cmds = data.get("hooks", {}).get("userPromptSubmit", [])
            data["hooks"]["userPromptSubmit"] = [c for c in cmds if c != _HOOK_COMMAND]
            self._settings_path.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, KeyError):
            pass

    def is_installed(self) -> bool:
        if not self._settings_path.exists():
            return False
        try:
            data = json.loads(self._settings_path.read_text())
            return _HOOK_COMMAND in data.get("hooks", {}).get("userPromptSubmit", [])
        except (json.JSONDecodeError, KeyError):
            return False
