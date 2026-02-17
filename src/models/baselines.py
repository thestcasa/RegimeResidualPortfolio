from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def momentum_12_1_scores(frame: pd.DataFrame) -> pd.Series:
    return frame["mom_252"].fillna(0.0)
