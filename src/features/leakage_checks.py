from __future__ import annotations

import pandas as pd


def assert_no_lookahead(features: pd.DataFrame, label_col: str = "target") -> None:
    """Basic sanity: no label timestamp should be earlier than feature month."""
    if "target_month" in features.columns:
        invalid = features[features["target_month"] <= features["month"]]
        if not invalid.empty:
            raise AssertionError("Detected lookahead alignment violations.")
