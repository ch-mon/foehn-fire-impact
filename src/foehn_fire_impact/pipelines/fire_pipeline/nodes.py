

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

def cleanse_fire_data(df_fire: pd.DataFrame) -> pd.DataFrame:
    df_fire.drop(columns=["ID cause reliability", "ID Cause", "ID exposition", "ID accuracy coordinates",
                          "ID accuracy end date", "ID accuracy start date", "ID current municipality",
                          "ID municipality", "definition", "ID definition", "ID fire"],
                 inplace=True)

    print(df_fire["start date"].dtype)
    logging.info(df_fire["start date"].dtype)
    
    return df_fire

def transform_datetime(df: pd.DataFrame) -> pd.DataFrame:

    # Convert to correct datetime
    df["start date (solar time)"] = pd.to_datetime(df["start date (solar time)"])
    df["end date (solar time)"] = pd.to_datetime(df["end date (solar time)"])

    # Initialize start and end fire datetimes
    df["start_date_min"] = pd.Timestamp.now()
    df["start_date_max"] = pd.Timestamp.now()
    df["end_date_min"] = pd.Timestamp.now()
    df["end_date_max"] = pd.Timestamp.now()

    # Get minimum and maximum start datetimes
    df["start_date_min"].loc[df["accuracy start date"] == "minute"] = df["start date (solar time)"]
    df["start_date_min"].loc[df["accuracy start date"] == "hour"] = df["start date (solar time)"].loc[df["accuracy start date"] == "hour"].apply(lambda dt: dt.replace(minute=0))
    df["start_date_max"].loc[df["accuracy start date"] == "minute"] = df["start date (solar time)"]
    df["start_date_max"].loc[df["accuracy start date"] == "hour"] = df["start date (solar time)"].loc[df["accuracy start date"] == "hour"].apply(lambda dt: dt.replace(minute=59))

    # Get minimum and maximum end datetimes
    df["end_date_min"].loc[df["accuracy end date"] == "minute"] = df["end date (solar time)"]
    df["end_date_min"].loc[df["accuracy end date"] == "hour"] = df["end date (solar time)"].loc[df["accuracy end date"] == "hour"].apply(lambda dt: dt.replace(minute=0))
    df["end_date_max"].loc[df["accuracy end date"] == "minute"] = df["end date (solar time)"]
    df["end_date_max"].loc[df["accuracy end date"] == "hour"] = df["end date (solar time)"].loc[df["accuracy end date"] == "hour"].apply(lambda dt: dt.replace(minute=59))

    # Calculate minimum and maximum duration
    df["duration_min"] = df["end_date_min"] - df["start_date_max"]
    df["duration_max"] = df["end_date_max"] - df["start_date_min"]
    df["duration_min"] = df["duration_min"].astype('timedelta64[m]')
    df["duration_max"] = df["duration_max"].astype('timedelta64[m]')

    # Drop durations which are negative
    df["duration_min"].loc[df["duration_min"] < pd.Timedelta(days=0, minutes=0)] = pd.NaT
    
    return df
