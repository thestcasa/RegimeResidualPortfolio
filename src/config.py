from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Thin wrapper around nested configuration dictionaries."""

    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)


def save_config(config: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.as_dict(), f, sort_keys=False)
