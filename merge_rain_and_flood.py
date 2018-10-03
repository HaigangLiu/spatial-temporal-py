import os
import pandas as pd
import numpy as np
from scipy import spatial

def location_based_matcher(main_df, query_df, varname):
    '''
    Used to merge two data files with different coordinates

    The main file is the original file, and the query df is
    the dataframe from where we borrow information.

    Example use case: the main_df is flood  while query df is about rain
    this function can help finding the nearest rain info for each observation
    in flood file and generate an addition column
    '''
    main_df_copy = main_df.copy()

    latitude = query_df.LATITUDE.values
    longitude = query_df.LONGITUDE.values
    kdtree = spatial.KDTree(list(zip(latitude, longitude)))

    #items to be queried
    pts_lat = main_df.LATITUDE.values
    pts_lon = main_df.LONGITUDE.values
    pts = list(zip(pts_lat, pts_lon))

    _, idx  = kdtree.query(pts, k=1)
    values_from_neighors = query_df.iloc[idx][varname].values
    main_df_copy[varname] = values_from_neighors

    return main_df_copy

def flood_and_rain_merge(main_df, start, end):
    '''
    merge the dataframe of different variables.
    the main dataframe contains, e.g. gage information and locations
    and for each location, we find the nearest rainfall information based on another dataframe.

    Args
    main_df (pandas dataframe): dataframe with location information indexed by latitude and longitude
    start (string): the start date. format should be 1990-09-09
    end (end): the end date. the format should be the same with start
    '''
    main_df_copy = main_df.copy()
    timestamps = pd.to_datetime(main_df_copy.DATE)
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    boolean_mask = (timestamps > start) & (timestamps < end)
    df_selected = main_df_copy[boolean_mask]

    container = []
    for date in pd.unique(df_selected.DATE):

        df_one_day = df_selected[df_selected.DATE == date]
        daily_prcp_file_name = ''.join(date.split('-')) + '.txt'
        file_dir = os.path.join('./rainfall_nws/', daily_prcp_file_name)

        try:
            prcp_for_that_day = pd.read_csv(file_dir, sep=' ')
            merged_rain_and_flood = location_based_matcher(df_one_day, prcp_for_that_day, varname='PRCP')
            print(f'finshed prcessoing for the data in {date}')
            container.append(merged_rain_and_flood)

        except FileNotFoundError:
            print(f'warning: the file for {date} is missing. Consider downloading it again')
    return pd.concat(container, ignore_index=True)

if __name__ == '__main__':
    #example
    base_file_for_gage = pd.read_csv('./data/flood_data_daily_beta.csv', index_col= 0)
    data = flood_and_rain_merge(base_file_for_gage, '2009-01-01', '2009-06-10')
    print(data.head())
