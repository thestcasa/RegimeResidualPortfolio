from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def hash_list(values: list[str]) -> str:
    joined = "|".join(sorted(values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def maybe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [maybe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: maybe_jsonable(v) for k, v in value.items()}
    return str(value)
