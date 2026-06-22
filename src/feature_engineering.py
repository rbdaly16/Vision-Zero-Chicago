"""Feature engineering utilities for Vision Zero Chicago crash analysis.

Functions to derive new features, classify fatalities, compute severity scores,
encode categorical variables, and merge datasets.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DAMAGE_MAPPING = {"OVER $1,500": 3, "$501 - $1,500": 2, "$500 OR LESS": 1}

INJURY_WEIGHTS = {
    "fatal": 7,
    "incapacitating": 3,
    "non_incapacitating": 2,
    "reported_not_evident": 1,
}

DEFAULT_CATEGORICAL_FEATURES = [
    "Weather Condition",
    "Roadway Surface Cond",
    "Road Defect",
    "Alignment",
    "Traffic Control Device",
    "Device Condition",
    "Crash Hour",
    "Trafficway Type",
    "Maneuver",
    "Vehicle Defect",
    "Lighting Condition",
    "First Crash Type",
    "Prim Contributory Cause",
    "Sec Contributory Cause",
]


def classify_fatality(injuries_fatal: int) -> str:
    """Return 'Fatal' if injuries_fatal > 0, otherwise 'Not Fatal'."""
    if injuries_fatal > 0:
        return "Fatal"
    return "Not Fatal"


def add_fatality_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Fatality Classification' column based on fatal injury count."""
    df = df.copy()
    df["Fatality Classification"] = df["Injuries Fatal"].apply(classify_fatality)
    return df


def compute_total_injured(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 'total injured' as sum of injury sub-categories."""
    df = df.copy()
    df["total injured"] = (
        df["Injuries Fatal"]
        + df["Injuries Incapacitating"]
        + df["Injuries Non Incapacitating"]
        + df["Injuries Reported Not Evident"]
    )
    return df


def compute_injury_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a weighted injury score from injury sub-categories.

    The score is a linear combination:
        7 * Fatal + 3 * Incapacitating + 2 * Non-Incapacitating + 1 * Reported Not Evident
    """
    df = df.copy()
    df["Injury Score"] = (
        df["Injuries Fatal"] * INJURY_WEIGHTS["fatal"]
        + df["Injuries Incapacitating"] * INJURY_WEIGHTS["incapacitating"]
        + df["Injuries Non Incapacitating"] * INJURY_WEIGHTS["non_incapacitating"]
        + df["Injuries Reported Not Evident"] * INJURY_WEIGHTS["reported_not_evident"]
    )
    return df


def map_damage_to_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Map the 'Damage' column to an ordinal encoding (1-3)."""
    df = df.copy()
    df["Damage Ode"] = df["Damage"].map(DAMAGE_MAPPING)
    return df


def compute_crash_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full crash score (injury score + damage component).

    Requires 'Injury Score' and 'Damage Ode' columns to already exist.
    """
    df = df.copy()
    df["Injury Score"] = df["Injury Score"] + df["Damage Ode"] * 3
    df["Crash Score"] = df["Injury Score"]
    return df


def merge_vehicles_crashes(
    vehicle_df: pd.DataFrame,
    crash_df: pd.DataFrame,
    on: str = "Crash Record Id",
) -> pd.DataFrame:
    """Inner-join vehicles and crashes, then deduplicate on the join key."""
    merged = vehicle_df.merge(crash_df, on=on, how="inner")
    merged.drop_duplicates(subset=on, inplace=True)
    return merged


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
) -> tuple:
    """One-hot encode the specified categorical columns.

    Returns:
        (encoded_df, encoder): The encoded DataFrame and fitted OneHotEncoder.
    """
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    encoded_array = ohe.fit_transform(df[categorical_columns])
    feature_names = ohe.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    return encoded_df, ohe


def calc_percentage(
    df: pd.DataFrame,
    columns: List[str],
    target_col: str = "Fatality Classification",
    target_value: str = "Fatal",
) -> pd.DataFrame:
    """Calculate the fatality percentage for each one-hot encoded column.

    For each column, computes the fraction of rows where the column equals 1
    and the target column equals the target value.

    Returns a DataFrame sorted by percentage in descending order.
    """
    percent_dict = {}
    for col in columns:
        col_present = df.loc[df[col] == 1]
        if len(col_present) == 0:
            percent_dict[col] = np.nan
        else:
            col_perc = len(
                df.loc[(df[col] == 1) & (df[target_col] == target_value)]
            ) / len(col_present)
            percent_dict[col] = col_perc

    percent_df = pd.DataFrame.from_dict(
        percent_dict, orient="index", columns=["Fatality Percentage"]
    )
    percent_df.sort_values(by="Fatality Percentage", ascending=False, inplace=True)
    return percent_df
