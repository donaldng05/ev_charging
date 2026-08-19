"""Sklearn learner registry: impute all, scale linear models only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chargeopt.models.learners import (
    LINEAR_LEARNERS,
    build_pipeline,
    fit_learner,
    predict_learner,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_linear_pipeline_includes_scaler() -> None:
    for name in LINEAR_LEARNERS:
        pipeline = build_pipeline(name, {"alpha": 1.0, "l1_ratio": 0.5}, seed=42)
        assert isinstance(pipeline, Pipeline)
        assert "scaler" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_tree_pipeline_omits_scaler() -> None:
    forest = build_pipeline(
        "random_forest",
        {"n_estimators": 8, "max_depth": 2, "min_samples_leaf": 1},
        seed=42,
    )
    extra = build_pipeline(
        "extra_trees",
        {"n_estimators": 8, "max_depth": 2, "min_samples_leaf": 1},
        seed=42,
    )
    boosting = build_pipeline(
        "hist_gradient_boosting",
        {"max_iter": 10, "max_depth": 2, "learning_rate": 0.1, "min_samples_leaf": 1},
        seed=42,
    )
    for pipeline in (forest, extra, boosting):
        assert "imputer" in pipeline.named_steps
        assert "scaler" not in pipeline.named_steps


def test_unknown_learner_raises() -> None:
    with pytest.raises(ValueError, match="unknown learner"):
        build_pipeline("svm", {"alpha": 1.0}, seed=42)


def test_fit_predict_is_finite_and_seeded() -> None:
    frame = _frame()
    first = fit_learner(
        frame,
        name="ridge",
        feature_columns=("x",),
        target_column="y",
        params={"alpha": 1.0},
        seed=7,
    )
    second = fit_learner(
        frame,
        name="ridge",
        feature_columns=("x",),
        target_column="y",
        params={"alpha": 1.0},
        seed=7,
    )
    left = predict_learner(frame, first, feature_columns=("x",))
    right = predict_learner(frame, second, feature_columns=("x",))
    np.testing.assert_allclose(left, right)
    assert np.isfinite(left).all()
