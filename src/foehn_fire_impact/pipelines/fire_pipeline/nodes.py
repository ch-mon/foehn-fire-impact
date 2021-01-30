# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

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
    df_fire.drop(columns=["ID cause reliability", "ID Cause"])
    
    return df_fire

def transform_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["start date (solar time)"] = pd.to_datetime(df["start date (solar time)"])
    df["end date (solar time)"] = pd.to_datetime(df["end date (solar time)"])

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
    # # Calculate the minimal and maximal possible duration of fire given the data

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    # Calculate lower and upper bound for start and end data
    df["start_date_min"] = pd.Timestamp.now()
    df["start_date_max"] = pd.Timestamp.now()
    df["end_date_min"] = pd.Timestamp.now()
    df["end_date_max"] = pd.Timestamp.now()

    df["start_date_min"].loc[df["accuracy start date"] =="minute"] = df["start date (solar time)"]
    df["start_date_min"].loc[df["accuracy start date"] =="hour"] = df["start date (solar time)"].loc[df["accuracy start date"] =="hour"].apply(lambda dt: dt.replace(minute = 0))
    # df["start_date_min"].loc[df["accuracy start date"] =="day"] = df["start date (solar time)"].loc[df["accuracy start date"] =="day"].apply(lambda dt: dt.replace(hour = 0, minute = 0))

    df["start_date_max"].loc[df["accuracy start date"] =="minute"] = df["start date (solar time)"]
    df["start_date_max"].loc[df["accuracy start date"] =="hour"] = df["start date (solar time)"].loc[df["accuracy start date"] =="hour"].apply(lambda dt: dt.replace(minute = 59))
    # df["start_date_max"].loc[df["accuracy start date"] =="day"] = df["start date (solar time)"].loc[df["accuracy start date"] =="day"].apply(lambda dt: dt.replace(hour = 23, minute = 59))

    df["end_date_min"].loc[df["accuracy end date"] =="minute"] = df["end date (solar time)"]
    df["end_date_min"].loc[df["accuracy end date"] =="hour"] = df["end date (solar time)"].loc[df["accuracy end date"] =="hour"].apply(lambda dt: dt.replace(minute = 0))
    # df["end_date_min"].loc[df["accuracy end date"] =="day"] = df["end date (solar time)"].loc[df["accuracy end date"] =="day"].apply(lambda dt: dt.replace(hour = 0, minute = 0))

    df["end_date_max"].loc[df["accuracy end date"] =="minute"] = df["end date (solar time)"]
    df["end_date_max"].loc[df["accuracy end date"] =="hour"] = df["end date (solar time)"].loc[df["accuracy end date"] =="hour"].apply(lambda dt: dt.replace(minute = 59))
    # df["end_date_max"].loc[df["accuracy end date"] =="day"] = df["end date (solar time)"].loc[df["accuracy end date"] =="day"].apply(lambda dt: dt.replace(hour = 23, minute = 59))

    df.head(10)

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    # Calculate differences
    df["duration_min"] = df["end_date_min"] - df["start_date_max"]
    df["duration_max"] = df["end_date_max"] - df["start_date_min"]

    df["duration_min"].loc[df["duration_min"] < pd.Timedelta(days = 0, minutes= 0)] = pd.NaT

    # Transforem i
    df["duration_min"]=df["duration_min"].astype('timedelta64[m]')
    df["duration_max"]=df["duration_max"].astype('timedelta64[m]')
    
    return df
