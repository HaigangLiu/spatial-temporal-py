from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy import spatial
from utility_functions import get_in_between_dates

class Merger:
    '''
    A class to merge two variables based on two data sources.
    For instance, there is rainfall data from charleston, but we only
    got gage data from columbia. Hence, we intend to merge the auxillary
    variable to the main dataframe.

    User decides which dataframe is the main one and which one is for query purpose,
    and can specify the start and the end date of the merging operation.
    '''
    def __init__(self, main_df, query_df, start, end, time_column=None,
        varname=None, lat_main=None, lon_main=None, lat_query=None, lon_query=None):

        self.main_df = main_df
        self.query_df = query_df
        self.list_of_dates = get_in_between_dates(start, end)
        self.varname = varname

        #column names
        self.varname = 'PRCP' if varname is None else varname
        self.lat_main = 'LATITUDE' if lat_main is None else lat_main
        self.lon_main = 'LONGITUDE' if lon_main is None else lon_main
        self.lat_query = 'LATITUDE' if lat_query is None else lat_query
        self.lon_query = 'LONGITUDE' if lon_query is None else lon_query
        self.time_column = 'DATE' if time_column is None else time_column

    def location_match_single_day(self, main_df, query_df):
        main_df_copy = main_df.copy()
        latitude = query_df[self.lat_query].values
        longitude = query_df[self.lon_query].values
        kdtree = spatial.KDTree(list(zip(latitude, longitude)))

        #items to be queried
        pts_lat = main_df[self.lat_main].values
        pts_lon = main_df[self.lon_main].values
        pts = list(zip(pts_lat, pts_lon))

        _, idx  = kdtree.query(pts, k=1)
        values_from_neighors = query_df.iloc[idx][self.varname].values
        main_df_copy[self.varname] = values_from_neighors
        return main_df_copy

    def run_single_date(self, date_):
        flood_info = self.main_df[self.main_df[self.time_column] == date_]
        rainfall_info = self.query_df[self.query_df[self.time_column] == date_]
        return self.location_match_single_day(flood_info, rainfall_info)

    def run(self):
        pool = Pool(processes=cpu_count())
        result = pool.map(self.run_single_date, self.list_of_dates)
        return pd.concat(list(result))


if __name__ == '__main__':
    rain = pd.read_csv('./demo/SC_20050101-20170627-19b7.txt', delimiter=" ")
    flood = pd.read_csv('./data/flood_data_daily_beta.csv', index_col=0, dtype={'SITENUMBER': str})

    merged_set = Merger(flood, rain, '2010-01-01', '2016-12-31').run()
    # merged_set.to_csv('./data/rainfall_and_flood_11.csv')
