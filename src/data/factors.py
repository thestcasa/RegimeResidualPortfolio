from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

FRENCH_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def fetch_ken_french_daily(cache_dir: str) -> pd.DataFrame:
    """Fetch and cache FF daily factors. Returns dataframe indexed by date."""
    cache_path = Path(cache_dir) / "factors" / "ff_daily.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    response = requests.get(FRENCH_DAILY_URL, timeout=30)
    response.raise_for_status()

    import zipfile

    zf = zipfile.ZipFile(BytesIO(response.content))
    member = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    raw = zf.read(member).decode("utf-8", errors="ignore").splitlines()

    rows: list[str] = []
    for line in raw:
        line = line.strip()
        if not line or not line[:8].isdigit():
            continue
        rows.append(line)

    df = pd.read_csv(BytesIO("\n".join(rows).encode("utf-8")), header=None)
    df.columns = ["date", "mkt_rf", "smb", "hml", "rf"]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    for c in ["mkt_rf", "smb", "hml", "rf"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df = df.dropna().set_index("date").sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def build_market_factor(benchmark_returns: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"mkt_rf": benchmark_returns})
    out.index.name = "date"
    return out
