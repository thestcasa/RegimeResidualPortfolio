from __future__ import annotations

from typing import Any


def get_universe(config: dict[str, Any]) -> list[str]:
    universe = config["data"]["universe"]["tickers"]
    return list(dict.fromkeys(universe))
