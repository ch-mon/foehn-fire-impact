import numpy as np


def combine_east_west_stations(df, east_station, west_station, final_station):
    foehn0_mask = (df[east_station] == 0.0) & (df[west_station] == 0.0)
    foehn1_mask = (df[east_station] == 1.0) | (df[west_station] == 1.0) & (df[east_station].notnull() & df[west_station].notnull())
    foehn2_mask = (df[east_station] == 2.0) | (df[west_station] == 2.0)

    df[final_station] = np.NaN
    df.loc[foehn0_mask, final_station] = 0.0
    df.loc[foehn1_mask, final_station] = 1.0
    df.loc[foehn2_mask, final_station] = 2.0

    df.drop([east_station, west_station], inplace=True, axis=1)

    return df
