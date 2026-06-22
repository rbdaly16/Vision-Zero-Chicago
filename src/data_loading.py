"""Utilities for loading Chicago crash data CSVs."""

import pandas as pd


DATA_DIR = "../data/localdata"

VEHICLE_FILE = "Traffic_Crashes_Vehicles.csv"
PEOPLE_FILE = "Traffic_Crashes_People.csv"
CRASH_FILE = "Traffic_Crashes_Crashes.csv"


def load_crash_data(data_dir=DATA_DIR):
    """Load the three crash data CSVs and return (vehicle_df, people_df, crash_df).

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the CSV files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        vehicle_df, people_df, crash_df
    """
    vehicle_df = pd.read_csv(
        f"{data_dir}/{VEHICLE_FILE}", low_memory=False
    )
    people_df = pd.read_csv(
        f"{data_dir}/{PEOPLE_FILE}", low_memory=False
    )
    crash_df = pd.read_csv(
        f"{data_dir}/{CRASH_FILE}", low_memory=False
    )
    return vehicle_df, people_df, crash_df
