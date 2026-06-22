"""Tests for src.data_cleaning module."""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    CRASH_DROP_COLUMNS,
    CRASH_REQUIRED_COLUMNS,
    PEOPLE_DROP_COLUMNS,
    PEOPLE_REQUIRED_COLUMNS,
    VEHICLE_KEEP_COLUMNS,
    VEHICLE_REQUIRED_COLUMNS,
    clean_crash_data,
    clean_people_data,
    clean_vehicle_data,
    load_csv,
    standardize_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_crash_df():
    """Minimal raw crash DataFrame mimicking the Chicago Data Portal schema."""
    return pd.DataFrame({
        "WORK_ZONE_I": ["Y", None, "N"],
        "WORK_ZONE_TYPE": ["A", "B", None],
        "DOORING_I": ["Y", "N", "N"],
        "WORKERS_PRESENT_I": ["Y", None, "N"],
        "PHOTOS_TAKEN_I": ["Y", "N", None],
        "STATEMENTS_TAKEN_I": ["Y", "N", "Y"],
        "RD_NO": ["RD001", "RD002", "RD003"],
        "REPORT_TYPE": ["A", "B", "C"],
        "CRASH_DATE_EST_I": ["Y", "N", "Y"],
        "INJURIES_TOTAL": [1, None, 3],
        "LATITUDE": [41.8, 41.9, None],
        "MOST_SEVERE_INJURY": ["FATAL", None, "NO INJURY"],
        "STREET_DIRECTION": ["N", "S", "E"],
        "BEAT_OF_OCCURRENCE": [111, 222, 333],
        "DAMAGE": ["OVER $1,500", "$501 - $1,500", "$500 OR LESS"],
        "CRASH_RECORD_ID": ["CR1", "CR2", "CR3"],
    })


@pytest.fixture
def raw_vehicle_df():
    """Minimal raw vehicle DataFrame."""
    data = {col: ["A", "B", "C"] for col in VEHICLE_KEEP_COLUMNS}
    data["VEHICLE_YEAR"] = [2020, np.nan, 2018]
    data["VEHICLE_USE"] = ["Personal", None, "Commercial"]
    data["FIRST_CONTACT_POINT"] = ["Front", "Rear", None]
    data["UNIT_TYPE"] = ["Driver", "Passenger", "Driver"]
    data["EXTRA_COL"] = [1, 2, 3]
    return pd.DataFrame(data)


@pytest.fixture
def raw_people_df():
    """Minimal raw people DataFrame."""
    base_cols = {col: ["X", "Y", "Z"] for col in PEOPLE_DROP_COLUMNS}
    base_cols.update({
        "AIRBAG_DEPLOYED": ["YES", None, "NO"],
        "EJECTION": ["NONE", "EJECTED", None],
        "INJURY_CLASSIFICATION": ["FATAL", "NO INJURY", "INCAPACITATING"],
        "VEHICLE_ID": [1, 2, 3],
        "SAFETY_EQUIPMENT": ["BELT", "NONE", "BELT"],
        "SEX": ["M", "F", "M"],
        "PERSON_ID": ["P1", "P2", "P3"],
    })
    return pd.DataFrame(base_cols)


# ---------------------------------------------------------------------------
# standardize_columns
# ---------------------------------------------------------------------------

class TestStandardizeColumns:
    def test_underscores_replaced(self):
        df = pd.DataFrame({"FOO_BAR": [1], "BAZ_QUX": [2]})
        result = standardize_columns(df)
        assert list(result.columns) == ["Foo Bar", "Baz Qux"]

    def test_title_case(self):
        df = pd.DataFrame({"HELLO": [1], "world": [2]})
        result = standardize_columns(df)
        assert list(result.columns) == ["Hello", "World"]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"FOO_BAR": [1]})
        standardize_columns(df)
        assert list(df.columns) == ["FOO_BAR"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = standardize_columns(df)
        assert result.empty


# ---------------------------------------------------------------------------
# clean_crash_data
# ---------------------------------------------------------------------------

class TestCleanCrashData:
    def test_drops_specified_columns(self, raw_crash_df):
        result = clean_crash_data(raw_crash_df)
        for col in CRASH_DROP_COLUMNS:
            title_col = col.title().replace("_", " ")
            assert title_col not in result.columns or col not in raw_crash_df.columns

    def test_drops_rows_with_missing_required(self, raw_crash_df):
        result = clean_crash_data(raw_crash_df)
        # Row index 1 (INJURIES_TOTAL=None, MOST_SEVERE_INJURY=None) and
        # row index 2 (LATITUDE=None) should be dropped
        assert len(result) == 1

    def test_column_names_standardized(self, raw_crash_df):
        result = clean_crash_data(raw_crash_df)
        for col in result.columns:
            assert "_" not in col
            assert col == col.title()

    def test_does_not_modify_original(self, raw_crash_df):
        original_cols = list(raw_crash_df.columns)
        original_len = len(raw_crash_df)
        clean_crash_data(raw_crash_df)
        assert list(raw_crash_df.columns) == original_cols
        assert len(raw_crash_df) == original_len

    def test_preserves_data_values(self, raw_crash_df):
        result = clean_crash_data(raw_crash_df)
        assert result["Injuries Total"].iloc[0] == 1

    def test_handles_missing_drop_columns_gracefully(self):
        df = pd.DataFrame({
            "INJURIES_TOTAL": [1],
            "LATITUDE": [41.8],
            "MOST_SEVERE_INJURY": ["FATAL"],
            "STREET_DIRECTION": ["N"],
            "BEAT_OF_OCCURRENCE": [111],
        })
        result = clean_crash_data(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# clean_vehicle_data
# ---------------------------------------------------------------------------

class TestCleanVehicleData:
    def test_selects_only_keep_columns(self, raw_vehicle_df):
        result = clean_vehicle_data(raw_vehicle_df)
        # "Extra Col" (from EXTRA_COL) should not be in result
        result_cols_upper = [c.upper().replace(" ", "_") for c in result.columns]
        assert "EXTRA_COL" not in result_cols_upper

    def test_drops_rows_missing_required(self, raw_vehicle_df):
        result = clean_vehicle_data(raw_vehicle_df)
        # Row with VEHICLE_USE=None (idx 1) and FIRST_CONTACT_POINT=None (idx 2)
        assert len(result) == 1

    def test_fills_missing_vehicle_year(self):
        df = pd.DataFrame({col: ["A"] for col in VEHICLE_KEEP_COLUMNS})
        df["VEHICLE_YEAR"] = [np.nan]
        df["VEHICLE_USE"] = ["Personal"]
        df["FIRST_CONTACT_POINT"] = ["Front"]
        df["UNIT_TYPE"] = ["Driver"]
        result = clean_vehicle_data(df)
        assert result["Vehicle Year"].iloc[0] == "Unknown"

    def test_column_names_standardized(self, raw_vehicle_df):
        result = clean_vehicle_data(raw_vehicle_df)
        for col in result.columns:
            assert col == col.title()

    def test_does_not_modify_original(self, raw_vehicle_df):
        original_len = len(raw_vehicle_df)
        clean_vehicle_data(raw_vehicle_df)
        assert len(raw_vehicle_df) == original_len


# ---------------------------------------------------------------------------
# clean_people_data
# ---------------------------------------------------------------------------

class TestCleanPeopleData:
    def test_drops_specified_columns(self, raw_people_df):
        result = clean_people_data(raw_people_df)
        for col in PEOPLE_DROP_COLUMNS:
            title_col = col.title().replace("_", " ")
            assert title_col not in result.columns

    def test_drops_rows_missing_required(self, raw_people_df):
        result = clean_people_data(raw_people_df)
        # Row 1 (AIRBAG_DEPLOYED=None) and row 2 (EJECTION=None) dropped
        assert len(result) == 1

    def test_column_names_standardized(self, raw_people_df):
        result = clean_people_data(raw_people_df)
        for col in result.columns:
            assert col == col.title()

    def test_does_not_modify_original(self, raw_people_df):
        original_len = len(raw_people_df)
        clean_people_data(raw_people_df)
        assert len(raw_people_df) == original_len

    def test_handles_missing_drop_columns_gracefully(self):
        df = pd.DataFrame({
            "AIRBAG_DEPLOYED": ["YES"],
            "EJECTION": ["NONE"],
            "INJURY_CLASSIFICATION": ["FATAL"],
            "VEHICLE_ID": [1],
            "SAFETY_EQUIPMENT": ["BELT"],
            "SEX": ["M"],
        })
        result = clean_people_data(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_loads_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        result = load_csv(str(csv_file))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")
