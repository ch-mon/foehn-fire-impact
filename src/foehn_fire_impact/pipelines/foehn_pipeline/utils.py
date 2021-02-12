import numpy as np


def combine_east_west_stations(df, east_station, west_station, final_station):
    '''
    Combine east and west station into one index with the following table.

    East | West | Desired outcome
    -----------------------------
    NA   | NA   | NA
    NA   | 0    | NA
    NA   | 1    | NA
    NA   | 2    | 2
    --
    0    | NA   | NA
    0    | 0    | 0
    0    | 1    | 1
    0    | 2    | 2
    --
    1    | NA   | NA
    1    | 0    | 1
    1    | 1    | 1
    1    | 2    | 2
    --
    2    | NA   | 2
    2    | 0    | 2
    2    | 1    | 2
    2    | 2    | 2
    -----------------------------

    :param df:
    :param east_station:
    :param west_station:
    :param final_station:
    :return:
    '''

    # Create masks for each case
    foehn0_mask = (df[east_station] == 0.0) & (df[west_station] == 0.0)
    foehn1_mask = ((df[east_station] == 1.0) | (df[west_station] == 1.0)) & ((df[east_station].notnull()) & (df[west_station].notnull()))
    foehn2_mask = (df[east_station] == 2.0) | (df[west_station] == 2.0)

    # Edit the correct entries. Order of these commands is important since masks are not mutually exclusive.
    df[final_station] = np.NaN
    df.loc[foehn0_mask, final_station] = 0.0
    df.loc[foehn1_mask, final_station] = 1.0
    df.loc[foehn2_mask, final_station] = 2.0

    # Drop now unnecessary columns
    df.drop([east_station, west_station], inplace=True, axis=1)

    return df
