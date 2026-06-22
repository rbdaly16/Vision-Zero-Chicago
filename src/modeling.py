"""Modeling utilities for Vision Zero Chicago crash prediction.

Functions to build, train, and evaluate classification models that predict
whether a crash is fatal, using Logistic Regression and Decision Tree
classifiers with SMOTE oversampling.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


FATAL_RECALL_SCORER = make_scorer(recall_score, pos_label="Fatal")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    test_size: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train/test sets."""
    return train_test_split(
        X, y, random_state=random_state, test_size=test_size
    )


def build_dummy_model(random_state: int = 42) -> DummyClassifier:
    """Build a baseline dummy classifier that always predicts the majority class."""
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def build_logistic_pipeline(
    random_state: int = 42,
    max_iter: int = 2000,
    smote_strategy: float = 0.3,
    class_weight: str = "balanced",
    penalty: Optional[str] = None,
    C: float = 1.0,
) -> Pipeline:
    """Build a SMOTE + Logistic Regression pipeline."""
    if penalty is None:
        logreg = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            C=np.inf,
            solver="lbfgs",
        )
    else:
        logreg = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            penalty=penalty,
            C=C,
            solver="saga" if penalty == "l1" else "lbfgs",
        )
    return Pipeline([
        ("smote", SMOTE(random_state=random_state, sampling_strategy=smote_strategy)),
        ("model", logreg),
    ])


def build_decision_tree_pipeline(
    random_state: int = 42,
    max_depth: int = 24,
    min_samples_split: int = 2500,
    smote_strategy: float = 0.3,
    class_weight: str = "balanced",
) -> Pipeline:
    """Build a SMOTE + Decision Tree pipeline."""
    clf = DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    return Pipeline([
        ("smote", SMOTE(random_state=random_state, sampling_strategy=smote_strategy)),
        ("model", clf),
    ])


def evaluate_recall(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    pos_label: str = "Fatal",
) -> float:
    """Compute recall score for the positive label."""
    y_pred = model.predict(X)
    return recall_score(y, y_pred, pos_label=pos_label)


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    n_jobs: int = -1,
) -> float:
    """Return mean cross-validated recall score."""
    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        scoring=FATAL_RECALL_SCORER,
        cv=cv,
        n_jobs=n_jobs,
    )
    return scores.mean()


def grid_search_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Run grid search over a SMOTE + Logistic Regression pipeline.

    Returns the fitted GridSearchCV object.
    """
    logreg = LogisticRegression(
        random_state=random_state,
        max_iter=2000,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=random_state, n_jobs=n_jobs)),
        ("model", logreg),
    ])

    if param_grid is None:
        param_grid = {
            "smote__sampling_strategy": [0.3, 0.4, 0.5],
            "model__C": np.logspace(-4, 4, 4).tolist(),
            "model__penalty": ["l1", None],
        }

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=FATAL_RECALL_SCORER,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    return search


def extract_coefficients(
    pipeline: Pipeline,
    feature_names: list,
) -> pd.DataFrame:
    """Extract and rank logistic regression coefficients from a fitted pipeline."""
    coefs = pipeline.named_steps["model"].coef_.flatten()
    coefs_negated = -coefs
    coefs_df = pd.DataFrame({
        "coefficients": coefs_negated,
        "features": feature_names,
    })
    coefs_df.sort_values(by="coefficients", ascending=False, inplace=True)
    return coefs_df.reset_index(drop=True)


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: list,
) -> pd.DataFrame:
    """Extract and rank decision tree feature importances from a fitted pipeline."""
    importances = pipeline.named_steps["model"].feature_importances_
    df = pd.DataFrame({
        "feature_importance": importances,
        "features": feature_names,
    })
    df.sort_values(by="feature_importance", ascending=False, inplace=True)
    return df.reset_index(drop=True)
