# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import os


def load_comprehensive_foehn_data(df_meteo_params, parameters):
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
            df = pd.read_csv(os.path.join(root, file), delimiter=";", usecols=["termin", parameter], encoding='latin1').rename(columns={"termin": "date", parameter: final_variable_name})

            # If dataframe does not contain rows, continue
            if len(df.index) == 0:
                continue

            # Convert columns and throw out 6-hourly values
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M00")

            ten_minute_mask = (df["date"].dt.minute == 10)
            amount_of_ten_minute_timestamps = ten_minute_mask.sum()
            index_start_of_10minute_measurements = ten_minute_mask.idxmax()

            if amount_of_ten_minute_timestamps != 0:

                df = df.loc[(index_start_of_10minute_measurements - 1):, :]

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

def load_older_data(parameters):

    df_final = pd.DataFrame(index=pd.date_range(start='1980-01-01 00:00', end='2019-12-31 23:59', freq="10min"))

    # Read in North foehn data
    north_foehn_list = os.listdir(parameters["older_north_foehn_data_path"])

    for location in north_foehn_list:
        logging.info(location)
        df = pd.read_csv(os.path.join(parameters["older_north_foehn_data_path"], location),
                         header=0,
                         names=["date", "foehn"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M")
        df.columns = ["date", location[0:3] + "_foehn"]
        df.dropna(subset=["date"], inplace=True)
        df_final = df_final.merge(df, on="date", how="left", validate="one_to_one")

    # TODO: Combine the MAE and MAW station into one MAG column
    MAG_dummy = df_final["MAE_foehn"].copy()
    MAG_dummy_mask = (df_final["MAW_foehn"] == 1.0)
    MAG_dummy.loc[MAG_dummy_mask] = 1.0
    MAG_dummy_mask = (df_final["MAE_foehn"] == 0.0) & (df_final["MAW_foehn"].isnull())
    MAG_dummy.loc[MAG_dummy_mask] = np.NaN

    df_final.loc[df_final["date"].dt.year < 2017, "MAG_foehn"] = MAG_dummy.loc[
        df_final["date"].dt.year < 2017].values

    # TODO: Combine the OTE and OTW station into one OTL column
    df_final["OTL_foehn"] = np.NaN
    OTL_dummy = df_final["OTE_foehn"].copy()
    OTL_dummy_mask = (df_final["OTW_foehn"] == 1.0)
    OTL_dummy.loc[OTL_dummy_mask] = 1.0
    OTL_dummy_mask = (df_final["OTE_foehn"] == 0.0) & (df_final["OTW_foehn"].isnull())
    OTL_dummy.loc[OTL_dummy_mask] = np.NaN

    df_final.loc[df_final["date"].dt.year < 2017, "OTL_foehn"] = OTL_dummy.loc[
        df_final["date"].dt.year < 2017].values

    df_final.drop(["MAW_foehn", "MAE_foehn", "OTW_foehn", "OTE_foehn"], inplace=True, axis=1)

    return df_final


def merge_old_and_new_foehn_data(df_new):
    ...

def cleanse_foehn_data(df):

    # Shift data from UTC format to Swiss time
    df["date"] = df["date"] + pd.Timedelta(hours=1)

    numerical_columns = df.drop(["date"], axis=1).columns
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] == "-", np.NaN)
    df[numerical_columns] = df[numerical_columns].astype(float)
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] > 2, np.NaN)
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] == 1.0, 0.0)
    df[numerical_columns] = df[numerical_columns].mask(df[numerical_columns] == 2.0, 1.0)
