"""Regression metrics for demand and energy models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    n = int(mask.sum())
    if n == 0:
        msg = "no paired observations for metrics"
        raise ValueError(msg)
    true = y_true.loc[mask].to_numpy(dtype=float)
    pred = y_pred.loc[mask].to_numpy(dtype=float)
    residual = true - pred
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "n": float(n),
    }


def metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grouped = predictions.groupby(["model", "split"], sort=True)
    for (model, split), group in grouped:
        stats = regression_metrics(group["target"], group["prediction"])
        rows.append({"model": str(model), "split": str(split), **stats})
    return pd.DataFrame(rows)


def test_mae_by_model(metrics: pd.DataFrame) -> dict[str, float]:
    test = metrics.loc[metrics["split"] == "test"]
    return {str(row["model"]): float(row["mae"]) for _, row in test.iterrows()}
