# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import os
from .utils import *


def load_comprehensive_foehn_data(df_meteo_params, parameters) -> pd.DataFrame:
    """
    Load the huge data dump from MeteoSwiss for all foehn stations in Switzerland.
    Bring it into processable columns and perform some safety checks (e.g. no 3 or 6 hourly values anymore)
    :param df_meteo_params: Table with necessary translation between variable names
    :param parameters: Dict with filepath to data delivery
    :return: Dataframe with all parameters and on a consistent 10-minute time axis.
    """

    meteo_param_mapping = dict(zip(df_meteo_params["PAR"].values, df_meteo_params["final_variable_name"].values))

    # Read in North foehn data
    dict_of_dicts = {}
    for root, dirs, files in os.walk(parameters["foehn_data_path"]):

        # Only look at lowest level which contains the files
        path_dirs = root.split("/")
        if len(path_dirs) != 8:
            continue

        # Extract period and parameter abbr. from filepath
        period, parameter = path_dirs[6], path_dirs[7]
        logging.info(f"Currently in period: {period}")
        logging.info(f"{parameter}: ({meteo_param_mapping[parameter]})")

        # Loop over all files in subdirectory
        for file in files:
            logging.info(file)

            # Get station abbreviation from filename and construct final feature name
            station_abbr = file[:3]
            final_variable_name = f"{station_abbr}_{meteo_param_mapping[parameter]}"

            # Read file
            df = pd.read_csv(os.path.join(root, file), delimiter=";", usecols=["termin", parameter],
                             encoding='latin1').rename(columns={"termin": "date", parameter: final_variable_name})

            # If dataframe does not contain rows, continue
            if len(df.index) == 0:
                continue

            # Convert columns to datetime
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M00")

            # Only allow values which have a spacing of 10 minutes (there are 3 and 6-hourly values in the dataset)
            ten_minute_mask = ((df["date"].shift(periods=(-1)) - df["date"]) == pd.Timedelta(minutes=10))
            if (df.iloc[-1, 0] - df.iloc[-2, 0]) == pd.Timedelta(minutes=10): # Fix last entry of mask due to shift side effect
                ten_minute_mask.iloc[-1] = True
            df = df.loc[ten_minute_mask, :]

            # Control if a previous period already exists for specific variable
            if final_variable_name not in dict_of_dicts.keys():
                dict_of_dicts[final_variable_name] = dict()

            dict_of_dicts[final_variable_name].update(dict(zip(df["date"].values, df[final_variable_name].values)))

    # Ensure a continuous dataframe and join dictionaries
    logging.info("Started joining")
    df_final = pd.DataFrame(index=pd.date_range(start='1980-01-01 00:00', end='2019-12-31 23:59', freq="10min"))
    for key in dict_of_dicts.keys():
        logging.info(key)
        df = pd.DataFrame.from_dict(dict_of_dicts[key], orient="index", columns=[key])
        df_final = df_final.join(df)

    df_final["date"] = df_final.index
    df_final.reset_index(drop=True, inplace=True)

    return df_final


def load_older_north_foehn_data(parameters) -> pd.DataFrame:
    """
    Load an older data delivery from MeteoSwiss which includes north foehn data that is not included in the newer data delivery.
    Their datawarhouse did not get filled with these values.
    :param parameters: Dict with filepath to data delivery
    :return: Dataframe with all parameters and on a consistent 10-minute time axis.
    """
    # Ensure a continuous dataframe
    df_final = pd.DataFrame(pd.date_range(start='1980-01-01 00:00', end='2019-12-31 23:59', freq="10min"), columns=["date"])

    # Read in North foehn data and merge into continuous dataframe
    north_foehn_list = os.listdir(parameters["older_north_foehn_data_path"])
    for location in north_foehn_list:
        logging.info(location)
        df = pd.read_csv(os.path.join(parameters["older_north_foehn_data_path"], location),
                         header=0,
                         names=["date", "foehn"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M")
        df.columns = ["date", location[0:3] + "_foehn"]
        df.dropna(subset=["date"], inplace=True)
        df_final = df_final.merge(df, on="date", how="left")

    # Combine the east and west stations in MAG and OTL
    df_final = combine_east_west_stations(df_final, "MAE_foehn", "MAW_foehn", "MAG_foehn")
    df_final = combine_east_west_stations(df_final, "OTE_foehn", "OTW_foehn", "OTL_foehn")

    return df_final


def merge_old_and_new_foehn_data(df_old, df_compr) -> pd.DataFrame:
    """
    Merge the old north foehn data from Matteo with the new foehn data from MeteoSwiss.
    Filter out the obvious wrong measurements.
    :param df_old: Old data delivery for north foehn data
    :param df_compr: Comprehensive second data delivery
    :return: Enriched dataframe which contains information from both data deliveries.
    """
    # If this assertion fails, then dataframes do not have the same datetime column and next steps are not possible
    assert (df_compr["date"] == df_old["date"]).all()

    # Fill every value which is NaN in comprehensive data with value from old north foehn data
    for col in df_old:
        logging.info(col)
        df_compr[col] = df_compr[col].fillna(df_old[col])

    # Drop the first year, since here none of the rows contains any value
    df_compr = df_compr.loc[df_compr["date"] >= np.datetime64("1981-01-01 00:00"), :].reset_index(drop=True)

    # Resolve obvious measurement errors/missing values
    df_compr.loc[df_compr["SBE_UU"] == 10000000.0, "SBE_UU"] = np.NaN
    df_compr.loc[df_compr["SBO_DD"] == 10000000.0, "SBO_DD"] = np.NaN

    # Filter geopotential heights (everything over 5000 on 850 hPa)
    for col in df_compr.filter(regex="Z850"):
        df_compr.loc[df_compr[col] > 5000.0, col] = np.NaN

    # Filter wind speed (everything over 150 km/h)
    for col in df_compr.filter(regex="FF$"):
        df_compr.loc[df_compr[col] > 150.0, col] = np.NaN

    # Filter wind speed gusts (everything over 200 km/h)
    for col in df_compr.filter(regex="FFX"):
        df_compr.loc[df_compr[col] > 200.0, col] = np.NaN

    return df_compr


def prepare_foehn_data_for_forest_fires(df) -> pd.DataFrame:
    # Shift data from UTC format to Swiss time
    df["date"] = df["date"] + pd.Timedelta(hours=1)

    numerical_columns = df.drop(["date"], axis=1).columns
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] == "-", np.NaN)
    df[numerical_columns] = df[numerical_columns].astype(float)
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] > 2, np.NaN)
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] == 1.0, 0.0)
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] == 2.0, 1.0)
