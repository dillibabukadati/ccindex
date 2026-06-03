from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class AgentAdapter(ABC):
    def __init__(self, project_root: Path):
        self.project_root = project_root

    @abstractmethod
    def install(self) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...

    @abstractmethod
    def is_installed(self) -> bool: ...
