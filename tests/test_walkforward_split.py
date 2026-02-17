import pandas as pd


def test_embargo_logic_example() -> None:
    months = pd.date_range("2018-01-31", periods=60, freq="M")
    min_train = 24
    val_months = 12
    embargo = 1
    i = min_train + val_months + embargo
    train_end = i - val_months - embargo
    train = months[:train_end]
    val = months[train_end : i - embargo]
    pred = months[i]
    assert train.max() < val.min()
    assert val.max() < pred
