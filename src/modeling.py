"""Utilities for building and evaluating classification models."""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def create_recall_scorer(pos_label="Fatal"):
    """Create a recall scorer for use with cross-validation and grid search.

    Parameters
    ----------
    pos_label : str
        The label to treat as the positive class.

    Returns
    -------
    sklearn scorer
        A scorer object compatible with sklearn's scoring parameter.
    """
    return make_scorer(recall_score, pos_label=pos_label)


def build_smote_pipeline(model, sampling_strategy=0.3, random_state=42):
    """Build an imbalanced-learn Pipeline with SMOTE resampling and a model.

    Parameters
    ----------
    model : sklearn estimator
        The classifier to use in the pipeline.
    sampling_strategy : float
        SMOTE sampling strategy (ratio of minority to majority).
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    imblearn.pipeline.Pipeline
        A pipeline chaining SMOTE and the classifier.
    """
    return Pipeline([
        ("smote", SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)),
        ("model", model),
    ])


def get_feature_coefficients(pipeline, feature_names):
    """Extract and sort model coefficients from a fitted pipeline.

    Assumes the pipeline has a step named 'model' with a ``coef_`` attribute
    (e.g. LogisticRegression).

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        A fitted pipeline containing a 'model' step.
    feature_names : list[str]
        List of feature names corresponding to the model coefficients.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'coefficients' and 'features' columns, sorted by
        coefficient magnitude (descending).
    """
    coef_values = pipeline.named_steps["model"].coef_.flatten()
    coefs_df = pd.DataFrame({
        "coefficients": -coef_values,
        "features": feature_names,
    })
    coefs_df.sort_values(by="coefficients", ascending=False, inplace=True)
    return coefs_df
