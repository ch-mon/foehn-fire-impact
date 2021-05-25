# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dypy.netcdf as dn
import scipy.spatial as spatial
import os
from .utils import *


def make_rain_dataset(df_stations, params):
    """
    Calculate all rain day for all stations.
    @param df_stations: Dataframe with all station coordinates.
    @param params: Dict with all global parameters.
    @return: Dataframe with rain amount and days for each station
    """
    
    # Obtain latitude and longitude coordiantes from simulation grid, transform coordinate system, and build them into cKDTree for fast lookup of nearest neigbors
    lat, lon = dn.read_var(os.path.join(params["rain_data_path"], "201501", "RhiresD_ch02.lonlat_20150101000000_20150131000000.nc"), variables=["lat", "lon"])
    points = np.array(np.meshgrid(lat, lon)).T.reshape(-1,2)
    x,y = decimalWSG84_to_LV3(lat = points[:, 0], lon=points[:,1])
    points = np.c_[x,y]
    point_tree = spatial.cKDTree(points)

    # Build a dictonary which contains all indices for the clostest grid points for each station
    index_dict = dict()
    for row in df_stations[["abbreviation", "x_LV03", "y_LV03"]].itertuples(index=False):
        indexes = point_tree.query_ball_point([row[1], row[2]], params["station_radius"])
        index_dict[row[0]] = indexes

    # Build a dictionary which contains dates and rain values from 1981-2019
    dict_of_dicts = dict()
    for dirpath, dirnames, filenames in os.walk(params["rain_data_path"]):  # Loop over all subdirectories
        print(filenames)
        for file in filenames: # Loop over all files in subdirs

            # Build a date index for a file
            start_date = file.split("_")[-2][:8]
            if int(start_date[:4])<1981:
                continue
            end_date = file.split("_")[-1].split(".")[0][:8]
            dates = pd.date_range(start=start_date, end=end_date, freq="1d")

            # Read gridded rain values
            try:
                rain, = dn.read_var(os.path.join(dirpath, file), variables=["RhiresD"])
            except Exception as e:
                print(e)  # Some files do not contain the correct variable name
                continue
            
            # Linearize grid
            rain = rain.reshape((rain.shape[0], -1))  
            print(rain.shape)
            assert len(dates)==rain.shape[0]

            # Loop over all stations and calculate rain average value for all days
            for station, index in index_dict.items():
                final_variable_name = f"{station}_rainavg"

                if final_variable_name not in dict_of_dicts.keys():
                    dict_of_dicts[final_variable_name] = dict()

                dict_of_dicts[final_variable_name].update(dict(zip(dates, rain[:, index].mean(axis=1))))

    # Ensure a continous dataframe from 1981-2019 and merge dict values into it
    df_rain = pd.DataFrame(index = pd.date_range(start="1981-01-01", end="2019-12-31", freq="1d"))
    for key in dict_of_dicts.keys():
        df = pd.DataFrame.from_dict(dict_of_dicts[key], orient="index", columns=[key])
        df_rain = df_rain.join(df)

    # Add date column
    df_rain["date"] = df_rain.index
    df_rain.reset_index(drop=True, inplace=True)
    
    # Define day as rainday if rainaverage is greater than 10 mm
    for col in df_rain.drop(columns=["date"]):
        station = col.split("_")[0]
        df_rain[f"{station}_rainday"] = (df_rain[col] > 10)
    
    return df_rain


