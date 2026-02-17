import pandas as pd
import pytest

from src.features.leakage_checks import assert_no_lookahead


def test_no_lookahead_pass() -> None:
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "target_month": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "target": [0.1, 0.2],
        }
    )
    assert_no_lookahead(df)


def test_no_lookahead_fail() -> None:
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2020-01-31"]),
            "target_month": pd.to_datetime(["2020-01-31"]),
            "target": [0.1],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_lookahead(df)
