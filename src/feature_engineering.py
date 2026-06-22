"""Utilities for feature engineering on crash data."""

import pandas as pd


DAMAGE_MAPPING = {"OVER $1,500": 3, "$501 - $1,500": 2, "$500 OR LESS": 1}

INJURY_WEIGHTS = {
    "fatal": 7,
    "incapacitating": 3,
    "non_incapacitating": 2,
    "reported_not_evident": 1,
}

DAMAGE_WEIGHT = 3


def compute_crash_score(df):
    """Compute the Crash Score for each row in a cleaned crash DataFrame.

    The Crash Score is a weighted linear combination of injury severity counts
    plus a damage component. Expects columns with standardized (title-case,
    space-separated) names.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned crash DataFrame with columns: 'Injuries Fatal',
        'Injuries Incapacitating', 'Injuries Non Incapacitating',
        'Injuries Reported Not Evident', and 'Damage'.

    Returns
    -------
    pd.DataFrame
        The DataFrame with added columns: 'Injury Score', 'Damage Ode',
        and 'Crash Score'.
    """
    df["Injury Score"] = (
        df["Injuries Fatal"] * INJURY_WEIGHTS["fatal"]
        + df["Injuries Incapacitating"] * INJURY_WEIGHTS["incapacitating"]
        + df["Injuries Non Incapacitating"] * INJURY_WEIGHTS["non_incapacitating"]
        + df["Injuries Reported Not Evident"] * INJURY_WEIGHTS["reported_not_evident"]
    )

    df["Damage Ode"] = df["Damage"].map(DAMAGE_MAPPING)
    df["Injury Score"] = df["Injury Score"] + df["Damage Ode"] * DAMAGE_WEIGHT
    df["Crash Score"] = df["Injury Score"]
    return df


def create_fatality_target(df):
    """Create a binary 'Fatality Classification' column.

    Rows with at least one fatal injury are labeled 'Fatal'; otherwise
    'Not Fatal'.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned crash DataFrame with an 'Injuries Fatal' column.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an added 'Fatality Classification' column.
    """
    df["Fatality Classification"] = df["Injuries Fatal"].apply(
        lambda x: "Fatal" if x > 0 else "Not Fatal"
    )
    return df


def calc_percentage(df, columns, target_col="Fatality Classification", target_val="Fatal"):
    """Calculate the fatality percentage for each binary-encoded column.

    For each column in *columns*, computes the proportion of rows where both
    the column equals 1 and the target column equals the target value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing binary feature columns and the target column.
    columns : list[str]
        List of binary-encoded feature column names to evaluate.
    target_col : str
        Name of the target column.
    target_val : str
        Value in the target column that counts as a positive case.

    Returns
    -------
    pd.DataFrame
        A single-column DataFrame indexed by feature name with the computed
        percentages, sorted descending.
    """
    percent_dict = {}
    for col in columns:
        total = len(df.loc[df[col] == 1])
        if total == 0:
            percent_dict[col] = 0.0
        else:
            percent_dict[col] = (
                len(df.loc[(df[col] == 1) & (df[target_col] == target_val)]) / total
            )

    percent_df = pd.DataFrame.from_dict(
        percent_dict, orient="index", columns=["Fatality Percentage"]
    )
    percent_df.sort_values(by="Fatality Percentage", ascending=False, inplace=True)
    return percent_df
