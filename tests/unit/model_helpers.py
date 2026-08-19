"""Shared tiny learner suites for unit tests."""

from __future__ import annotations

from chargeopt.config import LearnerSuite, load_config


def fast_learners(
    *,
    n_estimators: int = 20,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    max_iter: int = 20,
) -> LearnerSuite:
    base = load_config().models.demand.learners
    tree_search = {"n_estimators": [8, 12], "max_depth": [2], "min_samples_leaf": [1]}
    hgb_search = {
        "max_iter": [20],
        "max_depth": [2],
        "learning_rate": [0.1],
        "min_samples_leaf": [1],
    }
    return base.model_copy(
        update={
            "random_forest": base.random_forest.model_copy(
                update={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "search": base.random_forest.search.model_copy(update=tree_search),
                }
            ),
            "extra_trees": base.extra_trees.model_copy(
                update={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "search": base.extra_trees.search.model_copy(update=tree_search),
                }
            ),
            "hist_gradient_boosting": base.hist_gradient_boosting.model_copy(
                update={
                    "max_iter": max_iter,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "search": base.hist_gradient_boosting.search.model_copy(update=hgb_search),
                }
            ),
            "ridge": base.ridge.model_copy(
                update={"search": base.ridge.search.model_copy(update={"alpha": [0.1, 1.0]})}
            ),
            "elasticnet": base.elasticnet.model_copy(
                update={
                    "search": base.elasticnet.search.model_copy(
                        update={"alpha": [0.1, 1.0], "l1_ratio": [0.5]}
                    )
                }
            ),
        }
    )
