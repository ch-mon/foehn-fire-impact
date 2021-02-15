# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sci
import sklearn as sk
sns.set_style("whitegrid")

def map_fires_to_foehn(df_fires, df_foehn):
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
        new_features_dict.update(sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start=6))
        new_features_dict.update(sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start=12))

        rows_list.append(new_features_dict)

    return pd.concat([df_fires, pd.DataFrame(rows_list)], axis=1)


def sum_foehn_minutes_before_fire(fire, df_foehn, n_start, hours_before_start):

    fire_mask = slice((n_start - 6 * hours_before_start), (n_start - 1))
    foehn_values = df_foehn.loc[fire_mask, f'{fire["closest_station"]}_foehn']

    new_features_dict = {}
    if foehn_values.isnull().sum() == 0:
        foehn_minutes = foehn_values.sum() * 10

        if foehn_minutes == 0:
            TT_mean = np.NaN
            UU_mean = np.NaN
        else:
            TT_mean = df_foehn.loc[fire_mask, f'{fire["closest_station"]}_TT'].loc[foehn_values == 1.0].mean() - \
                      df_foehn.loc[fire_mask, f'{fire["closest_station"]}_TT'].loc[foehn_values == 0].mean()
            UU_mean = df_foehn.loc[fire_mask, f'{fire["closest_station"]}_UU'].loc[foehn_values == 1.0].mean() - \
                      df_foehn.loc[fire_mask, f'{fire["closest_station"]}_UU'].loc[foehn_values == 0].mean()

        new_features_dict[f"foehn_minutes_{hours_before_start}_hour_before"] = foehn_minutes
        new_features_dict[f"TT_mean_{hours_before_start}_hour_before"] = TT_mean
        new_features_dict[f"UU_mean_{hours_before_start}_hour_before"] = UU_mean
    else:
        new_features_dict[f"foehn_minutes_{hours_before_start}_hour_before"] = np.NaN
        new_features_dict[f"TT_mean_{hours_before_start}_hour_before"] = np.NaN
        new_features_dict[f"UU_mean_{hours_before_start}_hour_before"] = np.NaN

    return new_features_dict

def sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start):
    fire_mask = slice(n_start, (6 * hours_after_start + n_start - 1))
    foehn_values = df_foehn.loc[fire_mask, f'{fire["closest_station"]}_foehn']

    new_features_dict = {}
    # All foehn entries during the given period should be non-null
    if foehn_values.isnull().sum() == 0:
        foehn_minutes = foehn_values.sum() * 10
        FF_mean = df_foehn.loc[fire_mask, f'{fire["closest_station"]}_FF'].mean()
        FFX_mean = df_foehn.loc[fire_mask, f'{fire["closest_station"]}_FFX'].mean()

        new_features_dict[f"foehn_minutes_during_{hours_after_start}_hours_after_start_of_fire"] = foehn_minutes
        new_features_dict[f"FF_mean_during_{hours_after_start}_hours_after_start_of_fire"] = FF_mean
        new_features_dict[f"FFX_mean_during_{hours_after_start}_hours_after_start_of_fire"] = FFX_mean
    else:
        new_features_dict[f"foehn_minutes_during_{hours_after_start}_hours_after_start_of_fire"] = np.NaN
        new_features_dict[f"FF_mean_during_{hours_after_start}_hours_after_start_of_fire"] = np.NaN
        new_features_dict[f"FFX_mean_during_{hours_after_start}_hours_after_start_of_fire"] = np.NaN

    return new_features_dict

def add_control_variables(df):

    ## Fire regimes
    df["fire_regime"] = np.NaN

    # Winter anthropogenic
    mask = ((df["start_date_min"].dt.month <= 4) | (df["start_date_min"].dt.month == 12))
    df.loc[mask, "fire_regime"] = "Winter anthropogenic"

    # Summer natural
    mask = (df["start_date_min"].dt.month >= 5) & (df["start_date_min"].dt.month <= 11) & (
            df["cause"] == "lightning")
    df.loc[mask, "fire_regime"] = "Summer natural"

    # Summer anthropogenic
    mask = (df["start_date_min"].dt.month >= 5) & (df["start_date_min"].dt.month <= 11) & (
            df["cause"] != "unknown") & (df["cause"] != "lightning")
    df.loc[mask, "fire_regime"] = "Summer anthropogenic"

    ## South or North foehn feature
    north_foehn_stations = ["LUG", "OTL", "MAG", "COM", "GRO", "SBO", "PIO", "CEV", "ROB", "VIO"]
    south_foehn_stations = set(df["closest_station"].values) - set(north_foehn_stations)

    df["potential_foehn_species"] = np.NaN
    df.loc[df["closest_station"].isin(north_foehn_stations), "potential_foehn_species"] = "North foehn"
    df.loc[df["closest_station"].isin(south_foehn_stations), "potential_foehn_species"] = "South foehn"

    ## Decade feature
    df["decade"] = ""
    for year in range(1990, 2020 + 1, 10):
        decade_mask = (year - 10 <= df["start_date_min"].dt.year) & (df["start_date_min"].dt.year < year)
        df.loc[decade_mask, "decade"] = f"[{year - 10}, {year - 1}]"

    return df
