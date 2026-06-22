"""Utilities for cleaning and preparing crash data DataFrames."""

import pandas as pd


# Columns to drop from the crash DataFrame
CRASH_DROP_COLS = [
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

# Rows with NaN in these columns are dropped from the crash DataFrame
CRASH_REQUIRED_COLS = [
    "INJURIES_TOTAL",
    "LATITUDE",
    "MOST_SEVERE_INJURY",
    "STREET_DIRECTION",
    "BEAT_OF_OCCURRENCE",
]

# Columns to keep in the vehicle DataFrame
VEHICLE_KEEP_COLS = [
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

# Rows with NaN in these columns are dropped from the vehicle DataFrame
VEHICLE_REQUIRED_COLS = ["VEHICLE_USE", "FIRST_CONTACT_POINT", "UNIT_TYPE"]

# Columns to drop from the people DataFrame
PEOPLE_DROP_COLS = [
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

# Rows with NaN in these columns are dropped from the people DataFrame
PEOPLE_REQUIRED_COLS = [
    "AIRBAG_DEPLOYED",
    "EJECTION",
    "INJURY_CLASSIFICATION",
    "VEHICLE_ID",
    "SAFETY_EQUIPMENT",
    "SEX",
]


def standardize_columns(df):
    """Convert column names to title case and replace underscores with spaces.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be renamed in place.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with renamed columns.
    """
    df.columns = df.columns.str.title()
    df.columns = df.columns.str.replace("_", " ")
    return df


def clean_crash_df(df, drop_cols=None, required_cols=None):
    """Clean the crash DataFrame by dropping unnecessary columns and null rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw crash DataFrame.
    drop_cols : list[str] | None
        Columns to drop. Defaults to CRASH_DROP_COLS.
    required_cols : list[str] | None
        Columns that must be non-null. Defaults to CRASH_REQUIRED_COLS.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized column names.
    """
    if drop_cols is None:
        drop_cols = CRASH_DROP_COLS
    if required_cols is None:
        required_cols = CRASH_REQUIRED_COLS

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    df = standardize_columns(df)
    return df


def clean_vehicle_df(df, keep_cols=None, required_cols=None):
    """Clean the vehicle DataFrame by selecting columns and dropping null rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw vehicle DataFrame.
    keep_cols : list[str] | None
        Columns to keep. Defaults to VEHICLE_KEEP_COLS.
    required_cols : list[str] | None
        Columns that must be non-null. Defaults to VEHICLE_REQUIRED_COLS.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized column names.
    """
    if keep_cols is None:
        keep_cols = VEHICLE_KEEP_COLS
    if required_cols is None:
        required_cols = VEHICLE_REQUIRED_COLS

    available_keep = [c for c in keep_cols if c in df.columns]
    df = df[available_keep].copy()
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    df["VEHICLE_YEAR"] = df["VEHICLE_YEAR"].fillna("Unknown")
    df = standardize_columns(df)
    return df


def clean_people_df(df, drop_cols=None, required_cols=None):
    """Clean the people DataFrame by dropping unnecessary columns and null rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw people DataFrame.
    drop_cols : list[str] | None
        Columns to drop. Defaults to PEOPLE_DROP_COLS.
    required_cols : list[str] | None
        Columns that must be non-null. Defaults to PEOPLE_REQUIRED_COLS.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized column names.
    """
    if drop_cols is None:
        drop_cols = PEOPLE_DROP_COLS
    if required_cols is None:
        required_cols = PEOPLE_REQUIRED_COLS

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    df = standardize_columns(df)
    return df
