from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from app.models import AgentFeatureSettings


class AgentFeatureSettingsStore:
    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AgentFeatureSettings:
        if not self.settings_path.exists():
            return AgentFeatureSettings()
        try:
            payload = json.loads(self.settings_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return AgentFeatureSettings()
        try:
            return AgentFeatureSettings.model_validate(payload)
        except ValueError:
            return AgentFeatureSettings()

    def save(self, settings: AgentFeatureSettings) -> AgentFeatureSettings:
        lock_path = self.settings_path.with_suffix(".lock")
        with FileLock(str(lock_path)):
            self.settings_path.write_text(
                settings.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return settings
