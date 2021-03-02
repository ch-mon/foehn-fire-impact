import logging
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import time
from .utils import decimalWSG84_to_LV3, LV3_to_decimalWSG84, calc_distance


def cleanse_fire_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    First basic cleanse of the fire database
    :param df: Fire dataframe
    :return: Cleansed fire dataframe
    """
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

    logging.debug(len(df.index))
    return df

def fill_missing_coordinates(df):
    '''
    Obtain coordinates for municipality and fill the missing coordinates in dataframe
    :param df: Fire dataframe
    :return: Fire dataframe with filled coordinates
    '''

    # Identify where x and y are missing
    mask = df["coordinates_x"].isnull() | df["coordinates_y"].isnull()
    list_of_municipalities = sorted(list(set(df.loc[mask, "current municipality"])))
    logging.info(list_of_municipalities)

    # Retrieve locations for all municipalities via Nominatim API
    for municipality in list_of_municipalities:
        # Retrieve information
        time.sleep(1)
        geolocator = Nominatim(user_agent="MapSwissCitiesToLocation")
        location = geolocator.geocode(municipality, country_codes="CH")
        logging.info(f"{municipality} ({location.address}): ({location.latitude}, {location.longitude})")

        # Convert to LV3 coordinates
        x, y = decimalWSG84_to_LV3(lon = location.longitude, lat = location.latitude)

        # Complete missing entries in dataframe
        municipality_mask = (df["current municipality"] == municipality)
        df.loc[municipality_mask & mask, "coordinates_x"] = x
        df.loc[municipality_mask & mask, "coordinates_y"] = y

    # Also create WSG84 coordinates
    df["longitude"], df["latitude"] = LV3_to_decimalWSG84(x = df["coordinates_x"], y=df["coordinates_y"])

    return df


def calculate_closest_station(df_fire, df_stations, parameters):
    """
    Map each fire to the closest weather observation station
    :param df_fire: Fire dataframe
    :param df_stations: Datafraem with all stations where a foehn index is available
    :param parameters: Dict which holds the radius to consider around each weather station
    :return: Fire dataframe with now has the closest weather station associated.
    """

    # Drop Guetsch (which is just the crest station)
    df_stations = df_stations.loc[df_stations["abbreviation"] != "GUE", :].reset_index(drop=True)

    # Greedily search through all distances and find the minimum
    df_fire["closest_station"] = ""
    df_fire["closest_station_distance"] = 0.0
    for i in range(len(df_fire.index)):
        distances = calc_distance(df_fire.loc[i, "coordinates_x"], df_fire.loc[i, "coordinates_y"],
                                  df_stations["x_LV03"], df_stations["y_LV03"])

        n = distances.idxmin()
        dist = distances.min()
        # Only map fires which are in a given radius around one of the weather stations
        if dist < parameters["station_radius"]:
            df_fire.loc[i, "closest_station"] = df_stations.loc[n, "abbreviation"]
            df_fire.loc[i, "closest_station_distance"] = dist
        else:
            df_fire.loc[i, "closest_station"] = np.nan
            df_fire.loc[i, "closest_station_distance"] = np.nan

    # Drop all rows where the fire could not mapped to a station
    df_fire = df_fire.loc[df_fire["closest_station"].notnull(), :]
    logging.debug(f"{len(df_fire)} fires in dataset")
    return df_fire