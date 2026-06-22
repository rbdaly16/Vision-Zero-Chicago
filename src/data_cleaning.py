"""Data cleaning utilities for Vision Zero Chicago crash data.

Functions to load, clean, and standardize the three core datasets
(crashes, vehicles, people) sourced from the City of Chicago Data Portal.
"""

import pandas as pd


CRASH_DROP_COLUMNS = [
    "WORK_ZONE_I",
    "WORK_ZONE_TYPE",
    "DOORING_I",
    "WORKERS_PRESENT_I",
    "PHOTOS_TAKEN_I",
    "STATEMENTS_TAKEN_I",
    "RD_NO",
    "REPORT_TYPE",
    "CRASH_DATE_EST_I",
]

CRASH_REQUIRED_COLUMNS = [
    "INJURIES_TOTAL",
    "LATITUDE",
    "MOST_SEVERE_INJURY",
    "STREET_DIRECTION",
    "BEAT_OF_OCCURRENCE",
]

VEHICLE_KEEP_COLUMNS = [
    "CRASH_UNIT_ID",
    "CRASH_RECORD_ID",
    "CRASH_DATE",
    "UNIT_NO",
    "UNIT_TYPE",
    "VEHICLE_YEAR",
    "VEHICLE_USE",
    "VEHICLE_TYPE",
    "VEHICLE_DEFECT",
    "MANEUVER",
    "OCCUPANT_CNT",
    "AREA_00_I",
    "AREA_01_I",
    "AREA_02_I",
    "AREA_03_I",
    "AREA_04_I",
    "AREA_05_I",
    "AREA_06_I",
    "AREA_07_I",
    "AREA_08_I",
    "AREA_09_I",
    "AREA_10_I",
    "AREA_11_I",
    "AREA_12_I",
    "AREA_99_I",
    "FIRST_CONTACT_POINT",
]

VEHICLE_REQUIRED_COLUMNS = [
    "VEHICLE_USE",
    "FIRST_CONTACT_POINT",
    "UNIT_TYPE",
]

PEOPLE_DROP_COLUMNS = [
    "RD_NO",
    "CELL_PHONE_USE",
    "PEDPEDAL_ACTION",
    "PEDPEDAL_VISIBILITY",
    "PEDPEDAL_LOCATION",
    "SEAT_NO",
    "HOSPITAL",
    "EMS_AGENCY",
    "EMS_RUN_NO",
    "BAC_RESULT",
    "BAC_RESULT VALUE",
    "DRIVERS_LICENSE_STATE",
    "DRIVERS_LICENSE_CLASS",
    "CITY",
    "STATE",
    "ZIPCODE",
]

PEOPLE_REQUIRED_COLUMNS = [
    "AIRBAG_DEPLOYED",
    "EJECTION",
    "INJURY_CLASSIFICATION",
    "VEHICLE_ID",
    "SAFETY_EQUIPMENT",
    "SEX",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to title case and replace underscores with spaces."""
    df = df.copy()
    if len(df.columns) == 0:
        return df
    df.columns = df.columns.str.title().str.replace("_", " ")
    return df


def clean_crash_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the crashes DataFrame.

    Steps:
      1. Drop unnecessary columns.
      2. Drop rows with missing values in required columns.
      3. Standardize column names (title-case, spaces instead of underscores).
    """
    df = df.copy()
    cols_to_drop = [c for c in CRASH_DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    required = [c for c in CRASH_REQUIRED_COLUMNS if c in df.columns]
    if required:
        df.dropna(subset=required, inplace=True)

    df = standardize_columns(df)
    return df


def clean_vehicle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the vehicles DataFrame.

    Steps:
      1. Select only the relevant columns.
      2. Drop rows missing required fields.
      3. Fill missing vehicle years with 'Unknown'.
      4. Standardize column names.
    """
    df = df.copy()
    keep = [c for c in VEHICLE_KEEP_COLUMNS if c in df.columns]
    df = df[keep]

    required = [c for c in VEHICLE_REQUIRED_COLUMNS if c in df.columns]
    if required:
        df.dropna(subset=required, inplace=True)

    if "VEHICLE_YEAR" in df.columns:
        df["VEHICLE_YEAR"] = df["VEHICLE_YEAR"].fillna("Unknown")

    df = standardize_columns(df)
    return df


def clean_people_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the people DataFrame.

    Steps:
      1. Drop unnecessary columns.
      2. Drop rows missing required fields.
      3. Standardize column names.
    """
    df = df.copy()
    cols_to_drop = [c for c in PEOPLE_DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    required = [c for c in PEOPLE_REQUIRED_COLUMNS if c in df.columns]
    if required:
        df.dropna(subset=required, inplace=True)

    df = standardize_columns(df)
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(filepath)
