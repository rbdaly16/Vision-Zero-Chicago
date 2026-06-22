"""Tests for src.feature_engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    DAMAGE_MAPPING,
    INJURY_WEIGHTS,
    add_fatality_classification,
    calc_percentage,
    classify_fatality,
    compute_crash_score,
    compute_injury_score,
    compute_total_injured,
    encode_categorical_features,
    map_damage_to_ordinal,
    merge_vehicles_crashes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def crash_injury_df():
    """DataFrame with injury sub-category columns."""
    return pd.DataFrame({
        "Injuries Fatal": [1, 0, 0, 2],
        "Injuries Incapacitating": [0, 1, 0, 1],
        "Injuries Non Incapacitating": [2, 0, 1, 0],
        "Injuries Reported Not Evident": [0, 3, 0, 0],
    })


@pytest.fixture
def damage_df():
    return pd.DataFrame({
        "Damage": ["OVER $1,500", "$501 - $1,500", "$500 OR LESS", "UNKNOWN"],
    })


@pytest.fixture
def vehicle_df():
    return pd.DataFrame({
        "Crash Record Id": ["CR1", "CR2", "CR3"],
        "Maneuver": ["Straight", "Turn", "Straight"],
    })


@pytest.fixture
def crash_df():
    return pd.DataFrame({
        "Crash Record Id": ["CR1", "CR2", "CR4"],
        "Weather Condition": ["Clear", "Rain", "Snow"],
    })


@pytest.fixture
def categorical_df():
    return pd.DataFrame({
        "Color": ["Red", "Blue", "Red", "Green", "Blue"],
        "Size": ["S", "M", "L", "S", "L"],
    })


@pytest.fixture
def percentage_df():
    return pd.DataFrame({
        "feat_a": [1, 1, 0, 1, 0],
        "feat_b": [0, 1, 1, 0, 1],
        "Fatality Classification": ["Fatal", "Not Fatal", "Fatal", "Fatal", "Not Fatal"],
    })


# ---------------------------------------------------------------------------
# classify_fatality
# ---------------------------------------------------------------------------

class TestClassifyFatality:
    def test_fatal_positive(self):
        assert classify_fatality(1) == "Fatal"

    def test_fatal_multiple(self):
        assert classify_fatality(5) == "Fatal"

    def test_not_fatal_zero(self):
        assert classify_fatality(0) == "Not Fatal"

    def test_not_fatal_negative(self):
        assert classify_fatality(-1) == "Not Fatal"


# ---------------------------------------------------------------------------
# add_fatality_classification
# ---------------------------------------------------------------------------

class TestAddFatalityClassification:
    def test_adds_column(self, crash_injury_df):
        result = add_fatality_classification(crash_injury_df)
        assert "Fatality Classification" in result.columns

    def test_correct_classification(self, crash_injury_df):
        result = add_fatality_classification(crash_injury_df)
        expected = ["Fatal", "Not Fatal", "Not Fatal", "Fatal"]
        assert result["Fatality Classification"].tolist() == expected

    def test_does_not_modify_original(self, crash_injury_df):
        add_fatality_classification(crash_injury_df)
        assert "Fatality Classification" not in crash_injury_df.columns

    def test_all_fatal(self):
        df = pd.DataFrame({"Injuries Fatal": [1, 2, 3]})
        result = add_fatality_classification(df)
        assert all(result["Fatality Classification"] == "Fatal")

    def test_none_fatal(self):
        df = pd.DataFrame({"Injuries Fatal": [0, 0, 0]})
        result = add_fatality_classification(df)
        assert all(result["Fatality Classification"] == "Not Fatal")


# ---------------------------------------------------------------------------
# compute_total_injured
# ---------------------------------------------------------------------------

class TestComputeTotalInjured:
    def test_total_computation(self, crash_injury_df):
        result = compute_total_injured(crash_injury_df)
        assert "total injured" in result.columns
        expected = [3, 4, 1, 3]
        assert result["total injured"].tolist() == expected

    def test_all_zeros(self):
        df = pd.DataFrame({
            "Injuries Fatal": [0],
            "Injuries Incapacitating": [0],
            "Injuries Non Incapacitating": [0],
            "Injuries Reported Not Evident": [0],
        })
        result = compute_total_injured(df)
        assert result["total injured"].iloc[0] == 0

    def test_does_not_modify_original(self, crash_injury_df):
        compute_total_injured(crash_injury_df)
        assert "total injured" not in crash_injury_df.columns


# ---------------------------------------------------------------------------
# compute_injury_score
# ---------------------------------------------------------------------------

class TestComputeInjuryScore:
    def test_weighted_score(self, crash_injury_df):
        result = compute_injury_score(crash_injury_df)
        # Row 0: 1*7 + 0*3 + 2*2 + 0*1 = 11
        assert result["Injury Score"].iloc[0] == 11
        # Row 1: 0*7 + 1*3 + 0*2 + 3*1 = 6
        assert result["Injury Score"].iloc[1] == 6
        # Row 2: 0*7 + 0*3 + 1*2 + 0*1 = 2
        assert result["Injury Score"].iloc[2] == 2
        # Row 3: 2*7 + 1*3 + 0*2 + 0*1 = 17
        assert result["Injury Score"].iloc[3] == 17

    def test_all_zeros_score(self):
        df = pd.DataFrame({
            "Injuries Fatal": [0],
            "Injuries Incapacitating": [0],
            "Injuries Non Incapacitating": [0],
            "Injuries Reported Not Evident": [0],
        })
        result = compute_injury_score(df)
        assert result["Injury Score"].iloc[0] == 0

    def test_does_not_modify_original(self, crash_injury_df):
        compute_injury_score(crash_injury_df)
        assert "Injury Score" not in crash_injury_df.columns


# ---------------------------------------------------------------------------
# map_damage_to_ordinal
# ---------------------------------------------------------------------------

class TestMapDamageToOrdinal:
    def test_known_values(self, damage_df):
        result = map_damage_to_ordinal(damage_df)
        assert result["Damage Ode"].iloc[0] == 3
        assert result["Damage Ode"].iloc[1] == 2
        assert result["Damage Ode"].iloc[2] == 1

    def test_unknown_value_maps_to_nan(self, damage_df):
        result = map_damage_to_ordinal(damage_df)
        assert pd.isna(result["Damage Ode"].iloc[3])

    def test_does_not_modify_original(self, damage_df):
        map_damage_to_ordinal(damage_df)
        assert "Damage Ode" not in damage_df.columns


# ---------------------------------------------------------------------------
# compute_crash_score
# ---------------------------------------------------------------------------

class TestComputeCrashScore:
    def test_crash_score(self):
        df = pd.DataFrame({
            "Injury Score": [10, 5],
            "Damage Ode": [3, 1],
        })
        result = compute_crash_score(df)
        assert result["Crash Score"].iloc[0] == 10 + 3 * 3  # 19
        assert result["Crash Score"].iloc[1] == 5 + 1 * 3   # 8

    def test_injury_score_updated(self):
        df = pd.DataFrame({"Injury Score": [0], "Damage Ode": [2]})
        result = compute_crash_score(df)
        assert result["Injury Score"].iloc[0] == 6
        assert result["Crash Score"].iloc[0] == 6


# ---------------------------------------------------------------------------
# merge_vehicles_crashes
# ---------------------------------------------------------------------------

class TestMergeVehiclesCrashes:
    def test_inner_join(self, vehicle_df, crash_df):
        result = merge_vehicles_crashes(vehicle_df, crash_df)
        # Only CR1 and CR2 are common
        assert len(result) == 2
        assert set(result["Crash Record Id"]) == {"CR1", "CR2"}

    def test_columns_from_both(self, vehicle_df, crash_df):
        result = merge_vehicles_crashes(vehicle_df, crash_df)
        assert "Maneuver" in result.columns
        assert "Weather Condition" in result.columns

    def test_deduplication(self):
        v = pd.DataFrame({
            "Crash Record Id": ["CR1", "CR1", "CR2"],
            "Unit": [1, 2, 1],
        })
        c = pd.DataFrame({
            "Crash Record Id": ["CR1", "CR2"],
            "Weather": ["Clear", "Rain"],
        })
        result = merge_vehicles_crashes(v, c)
        assert len(result) == 2

    def test_empty_result_no_overlap(self):
        v = pd.DataFrame({"Crash Record Id": ["A"], "X": [1]})
        c = pd.DataFrame({"Crash Record Id": ["B"], "Y": [2]})
        result = merge_vehicles_crashes(v, c)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# encode_categorical_features
# ---------------------------------------------------------------------------

class TestEncodeCategoricalFeatures:
    def test_returns_dataframe_and_encoder(self, categorical_df):
        encoded_df, ohe = encode_categorical_features(
            categorical_df, ["Color", "Size"]
        )
        assert isinstance(encoded_df, pd.DataFrame)
        assert hasattr(ohe, "transform")

    def test_drops_first_category(self, categorical_df):
        encoded_df, _ = encode_categorical_features(
            categorical_df, ["Color", "Size"]
        )
        # OneHotEncoder with drop='first' should produce n_categories-1 cols
        # Color: 3 categories -> 2 cols, Size: 3 categories -> 2 cols => 4 total
        assert encoded_df.shape[1] == 4

    def test_binary_values(self, categorical_df):
        encoded_df, _ = encode_categorical_features(
            categorical_df, ["Color", "Size"]
        )
        assert set(encoded_df.values.flatten()).issubset({0.0, 1.0})

    def test_preserves_index(self, categorical_df):
        categorical_df.index = [10, 20, 30, 40, 50]
        encoded_df, _ = encode_categorical_features(
            categorical_df, ["Color", "Size"]
        )
        assert list(encoded_df.index) == [10, 20, 30, 40, 50]

    def test_single_column(self):
        df = pd.DataFrame({"A": ["x", "y", "x"]})
        encoded_df, _ = encode_categorical_features(df, ["A"])
        assert encoded_df.shape[1] == 1


# ---------------------------------------------------------------------------
# calc_percentage
# ---------------------------------------------------------------------------

class TestCalcPercentage:
    def test_correct_percentages(self, percentage_df):
        result = calc_percentage(percentage_df, ["feat_a", "feat_b"])
        # feat_a: 3 rows with val=1, 2 are Fatal -> 2/3
        assert pytest.approx(result.loc["feat_a", "Fatality Percentage"], rel=1e-6) == 2 / 3
        # feat_b: 3 rows with val=1, 1 is Fatal -> 1/3
        assert pytest.approx(result.loc["feat_b", "Fatality Percentage"], rel=1e-6) == 1 / 3

    def test_sorted_descending(self, percentage_df):
        result = calc_percentage(percentage_df, ["feat_a", "feat_b"])
        vals = result["Fatality Percentage"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_all_fatal(self):
        df = pd.DataFrame({
            "feat": [1, 1],
            "Fatality Classification": ["Fatal", "Fatal"],
        })
        result = calc_percentage(df, ["feat"])
        assert result.loc["feat", "Fatality Percentage"] == 1.0

    def test_none_fatal(self):
        df = pd.DataFrame({
            "feat": [1, 1],
            "Fatality Classification": ["Not Fatal", "Not Fatal"],
        })
        result = calc_percentage(df, ["feat"])
        assert result.loc["feat", "Fatality Percentage"] == 0.0

    def test_no_rows_with_feature_returns_nan(self):
        df = pd.DataFrame({
            "feat": [0, 0],
            "Fatality Classification": ["Fatal", "Not Fatal"],
        })
        result = calc_percentage(df, ["feat"])
        assert pd.isna(result.loc["feat", "Fatality Percentage"])

    def test_custom_target(self):
        df = pd.DataFrame({
            "feat": [1, 1, 1],
            "status": ["A", "B", "A"],
        })
        result = calc_percentage(
            df, ["feat"], target_col="status", target_value="A"
        )
        assert pytest.approx(result.loc["feat", "Fatality Percentage"], rel=1e-6) == 2 / 3
