

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

def cleanse_fire_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop superfluous columns
    df.drop(columns=["ID cause reliability", "ID Cause", "ID exposition", "ID accuracy coordinates",
                          "ID accuracy end date", "ID accuracy start date", "ID current municipality",
                          "ID municipality", "definition", "ID definition", "ID fire"],
                 inplace=True)

    # Drop rows where there are missing values in the date and accuracy variables
    df.dropna(subset=["start date (solar time)", "end date (solar time)", "accuracy start date", "accuracy end date"],
              inplace=True)

    # Drop rows where accuracy is not known to minute or hour accuracy
    df = df.loc[df["accuracy start date"].isin(["minute", "hour"]) &
             df["accuracy end date"].isin(["minute", "hour"]), :].copy()

    # NaN value in burned area means small burned area.
    # Thus replace zero and NaN values with 0.01 ha
    df.loc[df["total [ha]"].isnull(), "total [ha]"] = 0.01
    df.loc[df["total [ha]"] == 0.0, "total [ha]"] = 0.01

    # Rename two columns due to inconsistency with SwissTopo coordinate transform guide
    df.rename(columns={"coordinates x": "coordinates_y", "coordinates y": "coordinates_x"}, inplace=True)

    logging.debug(len(df.index))
    return df

def transform_datetime(df: pd.DataFrame) -> pd.DataFrame:

    # Initialize start and end fire datetimes
    df["start_date_min"] = pd.NaT
    df["start_date_max"] = pd.NaT
    df["end_date_min"] = pd.NaT
    df["end_date_max"] = pd.NaT

    # Get minimum and maximum start datetimes
    mask_minute = df["accuracy start date"] == "minute"
    mask_hour = df["accuracy start date"] == "hour"
    df.loc[mask_minute, "start_date_min"] = df.loc[mask_minute,"start date (solar time)"]
    df.loc[mask_hour, "start_date_min"] = df.loc[mask_hour, "start date (solar time)"].apply(lambda dt: dt.replace(minute=0))
    df.loc[mask_minute, "start_date_max"] = df.loc[mask_minute, "start date (solar time)"]
    df.loc[mask_hour, "start_date_max"] = df.loc[mask_hour, "start date (solar time)"].apply(lambda dt: dt.replace(minute=59))

    # Get minimum and maximum end datetimes
    mask_minute = df["accuracy end date"] == "minute"
    mask_hour = df["accuracy end date"] == "hour"
    df.loc[mask_minute, "end_date_min"] = df.loc[mask_minute,"end date (solar time)"]
    df.loc[mask_hour, "end_date_min"] = df.loc[mask_hour, "end date (solar time)"].apply(lambda dt: dt.replace(minute=0))
    df.loc[mask_minute, "end_date_max"] = df.loc[mask_minute, "end date (solar time)"]
    df.loc[mask_hour, "end_date_max"] = df.loc[mask_hour, "end date (solar time)"].apply(lambda dt: dt.replace(minute=59))

    # Calculate minimum and maximum duration
    df["duration_min"] = (df["end_date_min"] - df["start_date_max"]).dt.seconds/60
    df["duration_max"] = (df["end_date_max"] - df["start_date_min"]).dt.seconds/60

    # Drop durations which are negative
    df = df.loc[~((df["duration_min"] <= 0.0) | (df["duration_max"] <= 0.0)), :]

    # Drop unnecessary columns again
    df = df.drop(columns=["start_date_min", "start_date_max", "end_date_min", "end_date_max"])

    logging.debug(len(df.index))
    return df


