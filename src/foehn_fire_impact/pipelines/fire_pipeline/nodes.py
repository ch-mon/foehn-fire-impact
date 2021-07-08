import logging
import os

import geopandas
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import time
from .utils import decimalWSG84_to_LV3, LV3_to_decimalWSG84, calc_distance
from kedro.framework.session import get_current_session


def cleanse_fire_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    First basic cleanse of the fire database
    :param df: Fire dataframe
    :return: Cleansed fire dataframe
    """

    logging.info(f"{len(df.index)} fires in original dataset")
    # Drop superfluous columns
    # df.drop(columns=["ID cause reliability", "ID Cause", "ID exposition", "ID accuracy coordinates",
    #                       "ID accuracy end date", "ID accuracy start date", "ID current municipality",
    #                       "ID municipality", "definition", "ID definition", "ID fire"],
    #              inplace=True)

    # Drop rows where there are missing values in the date and accuracy variables
    # Solar time is Swiss winter time (ie. CEST)
    df.dropna(
        subset=["start date (solar time)", "end date (solar time)", "accuracy start date", "accuracy end date"],
        inplace=True
    )

    # Drop rows where accuracy is not known to minute or hour accuracy
    df = df.loc[df["accuracy start date"].isin(["minute", "hour"]) & df["accuracy end date"].isin(["minute", "hour"]), :].copy()

    # NaN value in burned area means small burned area.
    # Thus replace zero and NaN values with 0.01 ha
    df.loc[df["total [ha]"].isnull(), "total [ha]"] = 0.01
    df.loc[df["total [ha]"] == 0.0, "total [ha]"] = 0.01

    # Rename two columns due to inconsistency with SwissTopo coordinate transform guide
    df.rename(columns={"coordinates x": "coordinates_y", "coordinates y": "coordinates_x"}, inplace=True)

    logging.info(f"{len(df.index)} fires in dataset after filtering fires for certain start/end accuracy")
    return df


def transform_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform date values in dataframe and add fire length column. Also drop fires with negative duration.
    :param df: Fire dataframe
    :return: Fire dataframe with enriched date information
    """

    # Initialize start and end fire datetimes
    df["start_date_min"] = pd.NaT
    df["start_date_max"] = pd.NaT
    df["end_date_min"] = pd.NaT
    df["end_date_max"] = pd.NaT

    # Get minimum and maximum start datetimes
    mask_minute = df["accuracy start date"] == "minute"
    mask_hour = df["accuracy start date"] == "hour"
    df.loc[mask_minute, "start_date_min"] = df.loc[mask_minute, "start date (solar time)"]
    df.loc[mask_hour, "start_date_min"] = df.loc[mask_hour, "start date (solar time)"].apply(lambda dt: dt.replace(minute=0))
    df.loc[mask_minute, "start_date_max"] = df.loc[mask_minute, "start date (solar time)"]
    df.loc[mask_hour, "start_date_max"] = df.loc[mask_hour, "start date (solar time)"].apply(lambda dt: dt.replace(minute=59))

    # Get minimum and maximum end datetimes
    mask_minute = df["accuracy end date"] == "minute"
    mask_hour = df["accuracy end date"] == "hour"
    df.loc[mask_minute, "end_date_min"] = df.loc[mask_minute, "end date (solar time)"]
    df.loc[mask_hour, "end_date_min"] = df.loc[mask_hour, "end date (solar time)"].apply(lambda dt: dt.replace(minute=0))
    df.loc[mask_minute, "end_date_max"] = df.loc[mask_minute, "end date (solar time)"]
    df.loc[mask_hour, "end_date_max"] = df.loc[mask_hour, "end date (solar time)"].apply(lambda dt: dt.replace(minute=59))

    # Calculate minimum and maximum duration (in hours)
    df["duration_min"] = (df["end_date_min"] - df["start_date_max"]).dt.total_seconds()/3600
    df["duration_max"] = (df["end_date_max"] - df["start_date_min"]).dt.total_seconds()/3600

    # Drop fires which durations are negative, zero or more than 3 months long
    df = df.loc[~((df["duration_min"] <= 0) | (df["duration_max"] <= 0) | (df["duration_min"] > 3*24*30)), :]

    logging.info(f"{len(df.index)} fires in dataset after filtering fires with unrealistic duration")
    return df


def fill_missing_coordinates(df):
    """
    Obtain center coordinates for municipality and fill the missing coordinates in dataframe
    :param df: Fire dataframe
    :return: Fire dataframe with filled coordinates
    """

    # Identify where x and y are missing
    missing_mask = df["coordinates_x"].isnull() | df["coordinates_y"].isnull()
    list_of_municipalities = sorted(list(set(df.loc[missing_mask, "current municipality"])))
    logging.info("Imputing the following municipalities: " + str(list_of_municipalities))

    # Retrieve locations for all municipalities via Nominatim API
    for municipality in list_of_municipalities:
        # Retrieve information
        time.sleep(1)
        geolocator = Nominatim(user_agent="MapSwissCitiesToLocation")
        location = geolocator.geocode(municipality, country_codes="CH")
        logging.info(f"Imputing {municipality} ({location.address}): ({location.latitude}, {location.longitude})")

        # Convert to LV3 coordinates
        x, y = decimalWSG84_to_LV3(lon=location.longitude, lat=location.latitude)

        # Complete missing entries in dataframe
        municipality_mask = (df["current municipality"] == municipality)
        df.loc[municipality_mask & missing_mask, "coordinates_x"] = x
        df.loc[municipality_mask & missing_mask, "coordinates_y"] = y

    # Also create WSG84 coordinates
    df["longitude"], df["latitude"] = LV3_to_decimalWSG84(x=df["coordinates_x"], y=df["coordinates_y"])

    logging.info(f"{len(df.index)} fires in dataset after coordinate imputation")
    return df


def calculate_closest_station(df_fire, df_stations, parameters, respect_topography=True):
    """
    Map each fire to the closest weather observation station
    :param df_fire: Fire dataframe
    :param df_stations: Dataframe with all stations where a foehn index is available
    :param parameters: Dict with project global parameters
    :param respect_topography: Respect topography when mapping (True) or only draw a circle around each station (False)
    :return: Fire dataframe with the closest weather station.
    """

    # Get path to main project directory
    project_path = get_current_session().load_context().project_path

    # Read list of all stations which should be taken into consideration
    regions = parameters["regions"]
    stations = regions["southern_switzerland"] + regions["northern_switzerland"]

    if respect_topography:
        # Read shape files from manual mapping efforts. Simply update shapefiles, if a different mapping is required
        # For whatever reason, column names in shapefile are cut
        shapes = geopandas.read_file(
            os.path.join(project_path, "data", "01_raw", "station_regions", "region_station_new.shp")
        )

        # Only consider allowed stations
        shapes = shapes.loc[shapes["abbreviati"].isin(stations), :]

        # Define dictionary which contain station abbreviation and polygon(s) as key-values
        polygon_dict = dict(zip(shapes["abbreviati"], shapes["geometry"]))

        # Turn fire dataframe into GeoDataFrame. Make x and y coordinates a Point object
        df_fire = geopandas.GeoDataFrame(
            df_fire, geometry=geopandas.points_from_xy(df_fire["coordinates_y"], df_fire["coordinates_x"]),
            crs="EPSG:21781"
        )

        # Loop over all stations and check which fires belong to which station area
        for station, polygon in polygon_dict.items():
            df_fire.loc[df_fire["geometry"].within(polygon), "abbreviation"] = station

        # Delete unnecessary column again
        df_fire = pd.DataFrame(df_fire)
        del df_fire["geometry"]

        # Remove fires in valleys with very little foehn (might want to remove this when updating the shapefiles)
        df_fire = df_fire.loc[~((df_fire["abbreviation"] == "ROB") & (df_fire["longitude"] < 9.8)), :].reset_index(drop=True).copy()
        df_fire = df_fire.loc[~((df_fire["abbreviation"] == "CHU") & (df_fire["longitude"] < 9.35)), :].reset_index(drop=True).copy()

        # Calculate distance between station and fire
        df_fire["closest_station_distance"] = calc_distance(
            df_fire["coordinates_x"], df_fire["coordinates_y"],
            df_stations["x_LV03"], df_stations["y_LV03"]
        )

    else:
        # Drop Guetsch (which is just the crest station)
        df_stations = df_stations.loc[df_stations["abbreviation"] != "GUE", :].reset_index(drop=True)

        # Greedily search through all distances and find the minimum
        df_fire["abbreviation"] = ""
        df_fire["closest_station_distance"] = 0.0
        for i in df_fire.index:
            distances = calc_distance(
                df_fire.loc[i, "coordinates_x"], df_fire.loc[i, "coordinates_y"],
                df_stations["x_LV03"], df_stations["y_LV03"]
            )

            # Get indices and distance for closest station
            n = distances.idxmin()
            dist = distances.min()

            # Only map fires which are in a given radius around one of the weather stations
            if dist < parameters["station_radius"]:
                df_fire.loc[i, "abbreviation"] = df_stations.loc[n, "abbreviation"]
                df_fire.loc[i, "closest_station_distance"] = dist
            else:
                df_fire.loc[i, "abbreviation"] = np.nan
                df_fire.loc[i, "closest_station_distance"] = np.nan

    # Drop all rows where the fire could not mapped to a station. Filter for fires which are at allowed stations
    df_fire = df_fire.loc[df_fire["abbreviation"].notnull() & df_fire["abbreviation"].isin(stations), :]
    logging.info(f"{len(df_fire.index)} fires in dataset after mapping to stations")

    return df_fire
