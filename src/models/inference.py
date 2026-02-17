from __future__ import annotations

import numpy as np
import torch

from src.models.nn import MLPRegressor


def mc_dropout_predict(
    model: MLPRegressor,
    x: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.train()
    xb = torch.tensor(x, dtype=torch.float32)
    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            preds.append(model(xb).numpy())
    stack = np.vstack(preds)
    return stack.mean(axis=0), stack.std(axis=0)
