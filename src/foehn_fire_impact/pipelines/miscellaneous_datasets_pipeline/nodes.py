# -*- coding: utf-8 -*-
import logging

import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
import os
from .utils import *
from kedro.framework.session import get_current_session


def make_rain_dataset(df_stations, params):
    """
    Calculate daily spatial average of precipitation for all stations.
    Daily data goes from 0600 to 0600 UTC of next day.
    Also see documentation in documentation folder.
    @param df_stations: Dataframe with all station coordinates.
    @param params: Dict with all global parameters.
    @return: Dataframe with rain amount and days for each station
    """

    # Reduce stations dataframe only to selected stations
    # regions = params["regions"]
    # stations = regions["southern_switzerland"] + regions["northern_switzerland"]
    # df_stations = df_stations.loc[df_stations["abbreviation"].isin(stations), :].reset_index(drop=True)
    
    # Obtain latitude and longitude coordinates from simulation grid
    lat = nc.Dataset(os.path.join(params["rain_data_path"], "201501", "RhiresD_ch02.lonlat_20150101000000_20150131000000.nc"))["lat"][:]
    lon = nc.Dataset(os.path.join(params["rain_data_path"], "201501", "RhiresD_ch02.lonlat_20150101000000_20150131000000.nc"))["lon"][:]

    # Transform coordinate system
    points = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    x, y = decimalWSG84_to_LV3(lat=points[:, 0], lon=points[:, 1])

    # Build them into cKDTree for fast lookup of nearest neighbors
    points = np.c_[x, y]
    point_tree = cKDTree(data=points)

    # Build a dictionary which contains all indices for the closest grid points for each station
    index_dict = dict()
    for row in df_stations[["abbreviation", "x_LV03", "y_LV03"]].itertuples(index=False):
        indexes = point_tree.query_ball_point([row[1], row[2]], params["station_radius"])
        index_dict[row[0]] = indexes

    # Build a dictionary which contains dates and rain values from 1981-2019
    dict_of_dicts = dict()
    for dirpath, dirnames, filenames in os.walk(params["rain_data_path"]):  # Loop over all subdirectories
        logging.info("Reading following files: " + str(filenames))
        for file in filenames:  # Loop over all files in subdirs

            # Build a date index for a file
            start_date = file.split("_")[-2][:8]
            if int(start_date[:4]) < 1981:  # Do not consider files before 1981
                continue
            end_date = file.split("_")[-1].split(".")[0][:8]
            dates = pd.date_range(start=start_date, end=end_date, freq="1d")

            # Read gridded rain values
            try:
                rain = nc.Dataset(os.path.join(dirpath, file))["RhiresD"][:]
            except Exception as e:
                logging.info(e)  # Some files do not contain the correct variable name
                continue
            
            # Linearize grid
            rain = rain.reshape((rain.shape[0], -1))

            # Assert index and rain data have the same shape
            assert len(dates) == rain.shape[0]

            # Loop over all stations and calculate spatial rain average for all days
            for station, index in index_dict.items():
                final_variable_name = f"{station}_rainavg"

                # Add variable if not already there
                if final_variable_name not in dict_of_dicts.keys():
                    dict_of_dicts[final_variable_name] = dict()

                # Add daily rain average as key-value to dict
                dict_of_dicts[final_variable_name].update(dict(zip(dates, rain[:, index].mean(axis=1))))

    # Ensure a continuous dataframe from 1981-2019 and merge rain values into it
    df_rain = pd.DataFrame(index=pd.date_range(start="1981-01-01", end="2019-12-31", freq="1d"))
    for key in dict_of_dicts.keys():
        df = pd.DataFrame.from_dict(dict_of_dicts[key], orient="index", columns=[key])
        df_rain = df_rain.join(df)

    # Map missing value "--" to np.NaN for ROB
    df_rain["ROB_rainavg"] = df_rain["ROB_rainavg"].astype(float)

    # Add date column
    df_rain = df_rain.reset_index().rename(columns={"index": "date"})

    return df_rain


def load_fire_indices_data(df_stations, regions):
    project_path = get_current_session().load_context().project_path

    # Define mapping between stations in our file and the data delivery
    station_mapping = {"Bad Ragaz": "Bad_Ragaz",
                       "Güttingen": "Guttingen",
                       "Hörnli": "Hornli",
                       "Locarno / Monti": "Locarno_Monti",
                       "S. Bernardino": "S_Bernardino",
                       "St. Gallen": "St_Gallen",
                       "Wädenswil": "Wadenswil",
                       "Zürich / Fluntern": "Zurich_Fluntern"}

    # Create mapping from abbreviation to fullname (map above values, leave others as they are)
    abbrev_dict = dict(zip(df_stations["abbreviation"], df_stations["name"].map(station_mapping).fillna(df_stations["name"])))

    # Read all data from relevant stations into a dataframe
    df_raw = pd.DataFrame()
    for region, stations in regions.items():
        for station in stations:

            # Read the 19XX-2012 files
            try:
                filename = os.path.join(project_path, "data", "01_raw", "fireindice_1981_2012", f"{abbrev_dict[station]}_RESULT.csv")
                df_temp = pd.read_csv(filename)
                df_temp["abbreviation"] = station
                df_temp["region"] = region
                df_raw = pd.concat([df_raw, df_temp], axis=0)
                logging.info(f"(1980-2012) Joined {station}")
            except FileNotFoundError:
                logging.info(f"(1980-2012) {station} does not exist")

            # Read the 2000-2018 files
            try:
                filename = os.path.join(project_path, "data", "01_raw", "fireindice_2000_2018", f"{station}_RESULT.csv")
                df_temp = pd.read_csv(filename)
                df_temp["abbreviation"] = station
                df_temp["region"] = region
                df_raw = pd.concat([df_raw, df_temp], axis=0)
                logging.info(f"(2000-2018) Joined {station} ")
            except FileNotFoundError:
                logging.info(f"(2000-2018) {station} does not exist")

    # Convert date string to data
    df_raw["DateYYYYMMDD"] = pd.to_datetime(df_raw["DateYYYYMMDD"], format='%Y%m%d')

    # Ensure a consistent time axis from beginning 1981 until end 2018
    time_axis = pd.DataFrame({"DateYYYYMMDD": pd.date_range(start="1981-01-01", end="2018-12-31", freq="1D")})

    # Join fire indice data onto time axis and remove duplicates due to overlapping datasets (keep value from first dataset)
    df_full = pd.merge(time_axis, df_raw, on="DateYYYYMMDD", how="left")
    df_full = df_full.loc[~df_full.duplicated(subset=["DateYYYYMMDD", "abbreviation"], keep="first"), :]
    df_full = df_full.rename(columns={"DateYYYYMMDD": "date"})

    df_full = df_full.drop(columns=["Date", "year", "month", "day"])

    return df_full

