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