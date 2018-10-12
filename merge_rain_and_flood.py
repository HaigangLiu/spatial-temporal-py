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

    User decides which dataframe is the main one and which one is for query purpose, and can specify the start and the end date of the merging operation.
    '''


    def __init__(self,  main_df, query_df, start, end, varname='PRCP'):
        self.main_df = main_df
        self.query_df = query_df
        self.list_of_dates = get_in_between_dates(start, end)
        self.varname = varname

    def location_match_single_day(self, main_df, query_df):
        main_df_copy = main_df.copy()
        latitude = query_df.LATITUDE.values
        longitude = query_df.LONGITUDE.values
        kdtree = spatial.KDTree(list(zip(latitude, longitude)))

        #items to be queried
        pts_lat = main_df.LATITUDE.values
        pts_lon = main_df.LONGITUDE.values
        pts = list(zip(pts_lat, pts_lon))

        _, idx  = kdtree.query(pts, k=1)
        values_from_neighors = query_df.iloc[idx][self.varname].values
        main_df_copy[self.varname] = values_from_neighors
        return main_df_copy

    def select_date(self, date_):
        flood_info = self.main_df[self.main_df.DATE == date_]
        rainfall_info = self.query_df[self.query_df.DATE == date_]
        return self.location_match_single_day(flood_info, rainfall_info)

    def run(self):
        pool = Pool(processes=cpu_count())
        result = pool.map(self.select_date, self.list_of_dates)
        return pd.concat(list(result))

if __name__ == '__main__':
    import pickle
    with open('./data/cached_combined_file.pickle', 'rb') as pickle_file:
        rain = pickle.load(pickle_file)
    flood = pd.read_csv('./data/flood_data_daily_beta.csv', index_col=0, dtype={'SITENUMBER': str})

    hanlder = Merger(flood, rain, '2007-01-01', '2017-12-31')
    merged_dataset = hanlder.run()
    merged_dataset.to_csv('./data/rainfall_and_flood_11.csv')
