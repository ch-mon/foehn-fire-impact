# -*- coding: utf-8 -*-
import pandas as pd
from .utils import *


def map_fires_to_foehn(df_fires, df_foehn, regions):
    """
    Map all forest fires to to corresponding time period in foehn dataframe and check for foehn occurrence and strength.
    :param df_fires: Fire dataframe
    :param df_foehn: Foehn dataframe
    :return: Fire dataframe with mapped foehn values
    """

    # Only allow fires which are in the stations defined in the configs.
    stations = []
    for reg, sta in regions.items():
        stations.extend(sta)
    df_fires = df_fires.loc[df_fires["abbreviation"].isin(stations), :].reset_index(drop=True)

    # Loop over all fires
    rows_list = []
    for index, fire in df_fires.iterrows():
        new_features_dict = {}

        # Get start and end index in foehn dataframe for given forest fire
        n_start = (df_foehn["date"] <= fire["start_date_min"]).idxmin()
        # n_end = (df_foehn["date"] < fire["end_date_max"]).idxmin()

        # Foehn minutes before forest fire
        new_features_dict.update(sum_foehn_minutes_before_fire(fire, df_foehn, n_start, hours_before_start=24))
        new_features_dict.update(sum_foehn_minutes_before_fire(fire, df_foehn, n_start, hours_before_start=48))

        # Foehn minutes during starting hours of the fire
        new_features_dict.update(sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start=2))
        new_features_dict.update(sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start=6))
        new_features_dict.update(sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start=12))

        rows_list.append(new_features_dict)

    return pd.concat([df_fires, pd.DataFrame(rows_list)], axis=1)


def add_control_variables(df, cause, regions):
    """
    Add control variables for fire regime, foehn type and decade.
    :param df: Fire dataframe
    :return: Fire dataframe with control variables
    """

    ## Fire regimes
    df["fire_regime"] = np.NaN

    # Winter anthropogenic
    mask = ((df["start_date_min"].dt.month <= 4) | (df["start_date_min"].dt.month == 12))
    df.loc[mask, "fire_regime"] = "Winter anthropogenic"

    # Summer natural
    mask = (df["start_date_min"].dt.month >= 5) & (df["start_date_min"].dt.month <= 11) & (
            df["cause"] == "lightning")
    df.loc[mask, "fire_regime"] = "Summer natural"
    print((df["fire_regime"] == "Summer natural").sum())

    # Summer anthropogenic
    mask = (df["start_date_min"].dt.month >= 5) & (df["start_date_min"].dt.month <= 11) & (
            df["cause"] != "unknown") & (df["cause"] != "lightning")
    df.loc[mask, "fire_regime"] = "Summer anthropogenic"

    df = pd.merge(df.drop(columns="ID Cause"), cause, on="ID fire", how="left", validate="one_to_one")
    df.loc[df["ID Cause"] == 2.0, "fire_regime"] = "Summer natural"
    df.loc[df["ID Cause"] == 1.0, "fire_regime"] = "Summer anthropogenic"

    # Remove summer natural fires since those cannot be influenced by foehn during ignition (due to lightning)
    df = df.loc[df["fire_regime"] != "Summer natural", :].reset_index(drop=True).copy()
    print(len(df))

    ## South or North foehn feature
    df.loc[df["abbreviation"].isin(regions["southern_switzerland"]), "potential_foehn_species"] = "North foehn"
    df.loc[df["abbreviation"].isin(regions["northern_switzerland"]), "potential_foehn_species"] = "South foehn"

    ## Decade feature
    for year in range(1990, 2020 + 1, 10):
        decade_mask = (year - 10 <= df["start_date_min"].dt.year) & (df["start_date_min"].dt.year < year)
        df.loc[decade_mask, "decade"] = f"[{year - 10}, {year - 1}]"

    # Remove fires in valleys with very little foehn
    df = df.loc[~((df["abbreviation"] == "ROB") & (df["longitude"] < 9.8)), :].reset_index(drop=True).copy()
    df = df.loc[~((df["abbreviation"] == "CHU") & (df["longitude"] < 9.35)), :].reset_index(drop=True).copy()
    print(len(df))

    return df
