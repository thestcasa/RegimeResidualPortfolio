from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.nn import MLPRegressor


@dataclass
class TrainArtifacts:
    model: MLPRegressor
    scaler: StandardScaler
    val_loss: float


def _loss_fn(name: str) -> nn.Module:
    if name.lower() == "mae":
        return nn.L1Loss()
    return nn.HuberLoss(delta=1.0)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> TrainArtifacts:
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)

    model = MLPRegressor(x_train.shape[1], cfg["model"]["hidden_dims"], cfg["model"]["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["lr"], weight_decay=cfg["model"]["weight_decay"])
    loss_fn = _loss_fn(cfg["model"]["loss"])

    train_ds = TensorDataset(torch.tensor(x_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=cfg["model"]["batch_size"], shuffle=True)

    best_loss = float("inf")
    best_state = None
    patience = 0
    for _ in range(cfg["model"]["epochs"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(x_val_s, dtype=torch.float32))
            val_loss = float(loss_fn(val_pred, torch.tensor(y_val, dtype=torch.float32)).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= cfg["model"]["early_stopping_patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainArtifacts(model=model, scaler=scaler, val_loss=best_loss)
