"""Regression metrics for demand and energy models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

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


def learned_beats_baselines(
    test_mae: dict[str, float],
    *,
    learners: Sequence[str],
    baselines: Sequence[str],
) -> dict[str, bool]:
    baseline_maes = [test_mae[name] for name in baselines if name in test_mae]
    beats: dict[str, bool] = {}
    for name in learners:
        mae = test_mae.get(name)
        beats[name] = (
            mae is not None and bool(baseline_maes) and all(mae < other for other in baseline_maes)
        )
    return beats


def best_learned(test_mae: dict[str, float], learners: Sequence[str]) -> str | None:
    available = {name: test_mae[name] for name in learners if name in test_mae}
    if not available:
        return None
    return min(available.items(), key=lambda item: (item[1], item[0]))[0]


def error_slices_from_predictions(predictions: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "hour", "is_weekend", "era"}
    missing = [column for column in required if column not in demand.columns]
    if missing:
        msg = f"demand table missing slice columns: {missing}"
        raise ValueError(msg)
    meta = demand.loc[:, ["timestamp", "hour", "is_weekend", "era"]].drop_duplicates("timestamp")
    merged = predictions.merge(meta, on="timestamp", how="left")
    if merged[["hour", "is_weekend", "era"]].isna().any().any():
        msg = "prediction timestamps missing demand metadata"
        raise ValueError(msg)
    rows: list[dict[str, float | str | int | bool]] = []
    grouped = merged.groupby(["model", "split", "hour", "is_weekend", "era"], sort=True)
    for (model, split, hour, is_weekend, era), group in grouped:
        stats = regression_metrics(group["target"], group["prediction"])
        rows.append(
            {
                "model": str(model),
                "split": str(split),
                "hour": int(cast(int, hour)),
                "is_weekend": bool(is_weekend),
                "era": str(era),
                **stats,
            }
        )
    return pd.DataFrame(rows)
