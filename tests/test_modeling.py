"""Tests for src.modeling module."""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.modeling import (
    FATAL_RECALL_SCORER,
    build_decision_tree_pipeline,
    build_dummy_model,
    build_logistic_pipeline,
    cross_validate_model,
    evaluate_recall,
    extract_coefficients,
    extract_feature_importances,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures — small synthetic dataset for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create a small synthetic binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feat_1": rng.randn(n),
        "feat_2": rng.randn(n),
        "feat_3": rng.randn(n),
    })
    # Make ~20% "Fatal"
    y = pd.Series(
        ["Fatal" if v > 0.8 else "Not Fatal" for v in rng.rand(n)],
        name="Fatality Classification",
    )
    return X, y


@pytest.fixture
def split_synthetic(synthetic_data):
    X, y = synthetic_data
    return split_data(X, y, random_state=42, test_size=0.25)


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_returns_four_parts(self, synthetic_data):
        X, y = synthetic_data
        result = split_data(X, y)
        assert len(result) == 4

    def test_correct_sizes(self, synthetic_data):
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        assert len(X_train) == pytest.approx(len(X) * 0.7, abs=1)
        assert len(X_test) == pytest.approx(len(X) * 0.3, abs=1)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_reproducible(self, synthetic_data):
        X, y = synthetic_data
        r1 = split_data(X, y, random_state=0)
        r2 = split_data(X, y, random_state=0)
        pd.testing.assert_frame_equal(r1[0], r2[0])

    def test_different_seeds_differ(self, synthetic_data):
        X, y = synthetic_data
        r1 = split_data(X, y, random_state=0)
        r2 = split_data(X, y, random_state=99)
        assert not r1[0].equals(r2[0])


# ---------------------------------------------------------------------------
# build_dummy_model
# ---------------------------------------------------------------------------

class TestBuildDummyModel:
    def test_returns_dummy_classifier(self):
        model = build_dummy_model()
        assert isinstance(model, DummyClassifier)

    def test_strategy(self):
        model = build_dummy_model()
        assert model.strategy == "most_frequent"

    def test_predicts_majority(self, split_synthetic):
        X_train, X_test, y_train, y_test = split_synthetic
        model = build_dummy_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        majority = y_train.value_counts().idxmax()
        assert all(p == majority for p in preds)


# ---------------------------------------------------------------------------
# build_logistic_pipeline
# ---------------------------------------------------------------------------

class TestBuildLogisticPipeline:
    def test_returns_pipeline(self):
        pipe = build_logistic_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_smote_and_model(self):
        pipe = build_logistic_pipeline()
        assert "smote" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_model_is_logistic(self):
        pipe = build_logistic_pipeline()
        assert isinstance(pipe.named_steps["model"], LogisticRegression)

    def test_fit_and_predict(self, split_synthetic):
        X_train, X_test, y_train, y_test = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({"Fatal", "Not Fatal"})

    def test_custom_params(self):
        pipe = build_logistic_pipeline(
            max_iter=500, C=0.1, penalty="l1", class_weight="balanced"
        )
        model = pipe.named_steps["model"]
        assert model.max_iter == 500
        assert model.C == 0.1
        assert model.penalty == "l1"


# ---------------------------------------------------------------------------
# build_decision_tree_pipeline
# ---------------------------------------------------------------------------

class TestBuildDecisionTreePipeline:
    def test_returns_pipeline(self):
        pipe = build_decision_tree_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_model_is_tree(self):
        pipe = build_decision_tree_pipeline()
        assert isinstance(pipe.named_steps["model"], DecisionTreeClassifier)

    def test_custom_depth(self):
        pipe = build_decision_tree_pipeline(max_depth=10)
        assert pipe.named_steps["model"].max_depth == 10

    def test_fit_and_predict(self, split_synthetic):
        X_train, X_test, y_train, y_test = split_synthetic
        pipe = build_decision_tree_pipeline(
            max_depth=5, min_samples_split=10, smote_strategy=0.5
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# evaluate_recall
# ---------------------------------------------------------------------------

class TestEvaluateRecall:
    def test_recall_range(self, split_synthetic):
        X_train, X_test, y_train, y_test = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        pipe.fit(X_train, y_train)
        score = evaluate_recall(pipe, X_test, y_test)
        assert 0.0 <= score <= 1.0

    def test_perfect_recall(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series(["Fatal", "Not Fatal", "Fatal"])

        class PerfectModel:
            def predict(self, X):
                return ["Fatal", "Not Fatal", "Fatal"]

        score = evaluate_recall(PerfectModel(), X, y)
        assert score == 1.0

    def test_zero_recall(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series(["Fatal", "Not Fatal", "Fatal"])

        class BadModel:
            def predict(self, X):
                return ["Not Fatal", "Not Fatal", "Not Fatal"]

        score = evaluate_recall(BadModel(), X, y)
        assert score == 0.0


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidateModel:
    def test_returns_float(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        score = cross_validate_model(pipe, X_train, y_train, cv=3, n_jobs=1)
        assert isinstance(score, float)

    def test_score_range(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        score = cross_validate_model(pipe, X_train, y_train, cv=3, n_jobs=1)
        assert 0.0 <= score <= 1.0

    def test_dummy_low_recall(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        dummy = build_dummy_model()
        score = cross_validate_model(dummy, X_train, y_train, cv=3, n_jobs=1)
        # Dummy predicts majority -> recall for minority ("Fatal") should be 0
        assert score == 0.0


# ---------------------------------------------------------------------------
# extract_coefficients
# ---------------------------------------------------------------------------

class TestExtractCoefficients:
    def test_returns_dataframe(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        pipe.fit(X_train, y_train)
        result = extract_coefficients(pipe, list(X_train.columns))
        assert isinstance(result, pd.DataFrame)
        assert "coefficients" in result.columns
        assert "features" in result.columns

    def test_sorted_descending(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        pipe.fit(X_train, y_train)
        result = extract_coefficients(pipe, list(X_train.columns))
        coefs = result["coefficients"].tolist()
        assert coefs == sorted(coefs, reverse=True)

    def test_length_matches_features(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_logistic_pipeline(smote_strategy=0.5)
        pipe.fit(X_train, y_train)
        result = extract_coefficients(pipe, list(X_train.columns))
        assert len(result) == X_train.shape[1]


# ---------------------------------------------------------------------------
# extract_feature_importances
# ---------------------------------------------------------------------------

class TestExtractFeatureImportances:
    def test_returns_dataframe(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_decision_tree_pipeline(
            max_depth=5, min_samples_split=10, smote_strategy=0.5
        )
        pipe.fit(X_train, y_train)
        result = extract_feature_importances(pipe, list(X_train.columns))
        assert isinstance(result, pd.DataFrame)
        assert "feature_importance" in result.columns
        assert "features" in result.columns

    def test_sorted_descending(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_decision_tree_pipeline(
            max_depth=5, min_samples_split=10, smote_strategy=0.5
        )
        pipe.fit(X_train, y_train)
        result = extract_feature_importances(pipe, list(X_train.columns))
        imps = result["feature_importance"].tolist()
        assert imps == sorted(imps, reverse=True)

    def test_importances_sum_to_one(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_decision_tree_pipeline(
            max_depth=5, min_samples_split=10, smote_strategy=0.5
        )
        pipe.fit(X_train, y_train)
        result = extract_feature_importances(pipe, list(X_train.columns))
        assert pytest.approx(result["feature_importance"].sum(), abs=1e-6) == 1.0

    def test_length_matches_features(self, split_synthetic):
        X_train, _, y_train, _ = split_synthetic
        pipe = build_decision_tree_pipeline(
            max_depth=5, min_samples_split=10, smote_strategy=0.5
        )
        pipe.fit(X_train, y_train)
        result = extract_feature_importances(pipe, list(X_train.columns))
        assert len(result) == X_train.shape[1]


# ---------------------------------------------------------------------------
# FATAL_RECALL_SCORER
# ---------------------------------------------------------------------------

class TestFatalRecallScorer:
    def test_scorer_callable(self):
        assert callable(FATAL_RECALL_SCORER)
