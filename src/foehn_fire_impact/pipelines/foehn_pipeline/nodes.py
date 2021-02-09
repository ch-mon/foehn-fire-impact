# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

def load_comprehensive_foehn_data(df_foehnstations, df_parameters):
    df_foehnstations = pd.read_csv("/home/chmony/Documents/FoehnStrengthFireCorrelation/foehn_stations_switzerland.csv")
    df_foehnstations

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    df_parameters = pd.read_csv("/home/chmony/Documents/FoehnStrengthFireCorrelation/meteorological_parameters.csv")
    parameter_dict = dict(zip(df_parameters["PAR"].values, df_parameters["final_variable_name"]))
    parameter_dict["ppz850s0"] = "Z850"
    parameter_dict

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
    # # Ensure a continuos dataframe

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    df_final = pd.date_range(start='1980-01-01 00:00', end='2019-12-31 23:59', freq="10min")
    df_final = pd.DataFrame(index= df_final)

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
    # # Read in North foehn data

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    dict_of_dicts = {}

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    return df_final