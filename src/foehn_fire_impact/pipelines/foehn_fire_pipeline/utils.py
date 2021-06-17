import numpy as np


def sum_foehn_minutes_before_fire(fire, df_foehn, n_start, hours_before_start):
    """
    Aggregate foehn minutes, TT, and UU in the 24/48 hours before a fire ignition
    :param fire: One fire from fire dataframe
    :param df_foehn: Foehn dataframe
    :param n_start: Fire ignition
    :param hours_before_start: 24 or 48 hours
    :return: Dict: Foehn minutes, TT mean, UU mean for the specific fire and considered time period
    """
    mapped_station = fire["abbreviation"]

    # Get weather station values in foehn dataframe
    fire_mask = slice((n_start - 6 * hours_before_start), (n_start - 1))
    foehn_values = df_foehn.loc[fire_mask, f'{mapped_station}_foehn']

    new_features_dict = {}
    if foehn_values.isnull().sum() == 0:  # Only allow fires where all weather station values are known
        foehn_minutes = foehn_values.sum() * 10

        # If no foehn occurrence
        if foehn_minutes == 0:
            TT_mean = np.NaN
            UU_mean = np.NaN
        else:
            # Subtract the temperature and humidity for values which do not show foehn occurrence to distill foehn
            # temperature increase and humidity decrease
            TT_mean = df_foehn.loc[fire_mask, f'{mapped_station}_TT'].loc[foehn_values == 1.0].mean() - \
                      df_foehn.loc[fire_mask, f'{mapped_station}_TT'].loc[foehn_values == 0.0].mean()
            UU_mean = df_foehn.loc[fire_mask, f'{mapped_station}_UU'].loc[foehn_values == 1.0].mean() - \
                      df_foehn.loc[fire_mask, f'{mapped_station}_UU'].loc[foehn_values == 0.0].mean()

        new_features_dict[f"foehn_minutes_{hours_before_start}_hour_before"] = foehn_minutes
        # new_features_dict[f"foehn_mean_TT_{hours_before_start}_hour_before"] = TT_mean
        # new_features_dict[f"foehn_mean_UU_{hours_before_start}_hour_before"] = UU_mean
    # else:  # If not all values are known, set to NaN
    #     new_features_dict[f"foehn_minutes_{hours_before_start}_hour_before"] = np.NaN
    #     new_features_dict[f"TT_mean_{hours_before_start}_hour_before"] = np.NaN
    #     new_features_dict[f"UU_mean_{hours_before_start}_hour_before"] = np.NaN

    return new_features_dict


def sum_foehn_minutes_during_start_period_of_fire(fire, df_foehn, n_start, hours_after_start):
    """
    Aggregate foehn minutes, FF, and FFX in the 6/12 hours after a fire ignition
    :param fire: One fire from fire dataframe
    :param df_foehn: Foehn dataframe
    :param n_start: Fire ignition
    :param hours_after_start: 6 or 12 hours
    :return: Dict: Foehn minutes, TT mean, UU mean for the specific fire and considered time period
    """

    mapped_station = fire["abbreviation"]

    # Get values in foehn dataframe
    fire_mask = slice(n_start, (6 * hours_after_start + n_start - 1))
    foehn_values = df_foehn.loc[fire_mask, f'{mapped_station}_foehn']

    new_features_dict = {}
    if foehn_values.isnull().sum() == 0:  # Only allow fires where all weather station values are known
        # Sum all foehn values for occurrence and strength.
        # Select only entries which show foehn occurrence.
        foehn_minutes = foehn_values.sum() * 10
        FF_mean = df_foehn.loc[fire_mask, f'{mapped_station}_FF'].mean()
        FFX_values = df_foehn.loc[fire_mask, f'{mapped_station}_FFX']

        new_features_dict[f"foehn_minutes_during_{hours_after_start}_hours_after_start_of_fire"] = foehn_minutes
        new_features_dict[f"FF_mean_during_{hours_after_start}_hours_after_start_of_fire"] = FF_mean
        new_features_dict[f"FFX_mean_during_{hours_after_start}_hours_after_start_of_fire"] = FFX_values.mean()
        new_features_dict[f"FFX_q75_during_{hours_after_start}_hours_after_start_of_fire"] = FFX_values.quantile(0.75)
        new_features_dict[f"FFX_q90_during_{hours_after_start}_hours_after_start_of_fire"] = FFX_values.quantile(0.90)
    # else:  # If not all values are known, set to NaN
    #     new_features_dict[f"foehn_minutes_during_{hours_after_start}_hours_after_start_of_fire"] = np.NaN
    #     new_features_dict[f"FF_mean_during_{hours_after_start}_hours_after_start_of_fire"] = np.NaN
    #     new_features_dict[f"FFX_mean_during_{hours_after_start}_hours_after_start_of_fire"] = np.NaN

    return new_features_dict
