from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, forecast_horizon: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.head(output[:, -1, :])


@dataclass(slots=True)
class MetricBundle:
    mae: float
    rmse: float
    mape: float
    loss: float


def dataset_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_model(model: nn.Module, loader: DataLoader, target_scale: tuple[float, float], device: torch.device) -> MetricBundle:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            outputs = model(features)
            losses.append(float(criterion(outputs, target).cpu().item()))
            predictions.append(outputs.cpu().numpy())
            targets.append(target.cpu().numpy())
    if not predictions:
        return MetricBundle(mae=0.0, rmse=0.0, mape=0.0, loss=0.0)

    pred = np.concatenate(predictions, axis=0)
    truth = np.concatenate(targets, axis=0)
    mean, std = target_scale
    pred_actual = (pred * std) + mean
    truth_actual = (truth * std) + mean
    error = pred_actual - truth_actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.clip(np.abs(truth_actual), 1e-3, None)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    return MetricBundle(mae=mae, rmse=rmse, mape=mape, loss=float(np.mean(losses)))


def state_dict_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def ndarrays_to_state_dict(model: nn.Module, arrays: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    for key, array in zip(state_dict.keys(), arrays, strict=True):
        state_dict[key] = torch.tensor(array)
    model.load_state_dict(state_dict, strict=True)

