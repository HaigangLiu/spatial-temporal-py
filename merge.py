import pandas as pd
from scipy import spatial
from datetime import datetime

class Merger:
    def __init__(self, start, end):

        if type(start) == str:
            start = datetime.strptime(start, '%Y-%m-%d')
            end = datetime.strptime(end, '%Y-%m-%d')

        self.date_index = pd.date_range(start, end) #used to reindex
        self.start = start; self.end = end

    def add_source(self, src, grid_col='SITE', date_col='DATE', lat_col='LATITUDE', lon_col='LONGITUDE',info_col='PRCP'):

        source = src.set_index(grid_col)
        self.date_col_source = date_col
        self.lat_col_source = lat_col
        self.lon_col_source = lon_col
        self.info_col_source = info_col

        assert grid_col in src, f'KeyError. column {grid_col} not found'
        assert date_col in src, f'KeyError. column {date_col} not found'
        assert lat_col in src, f'KeyError. column {lat_col} not found'
        assert lon_col in src, f'KeyError. column {lon_col} not found'
        assert info_col in src, f'KeyError. column {info_col} not found'

        #make tree here
        self.summary_g = source.groupby([grid_col]).first()[[lat_col, lon_col]]
        source[date_col] = pd.to_datetime(source[date_col])
        self.source = source[(source[date_col] >= self.start) & (source
            [date_col]<= self.end)]
        self.kdtree = spatial.KDTree(self.summary_g[[lat_col, lon_col]].values)

    def add_target(self, df, site_col='SITENUMBER', date_col='DATE', lat_col='LATITUDE', lon_col='LONGITUDE'):

        self.target = df.set_index(site_col)
        self.site_col_target = site_col
        self.date_col_target = date_col
        self.lat_col_target = lat_col
        self.lon_col_target = lon_col

        assert site_col in df, f'KeyError. column {site_col} not found'
        assert date_col in df, f'KeyError. column {date_col} not found'
        assert lat_col in df, f'KeyError. column {lat_col} not found'
        assert lon_col in df, f'KeyError. column {lon_col} not found'

        self.summary_t = self.target.groupby(site_col).first()[[lat_col, lon_col]]
        self.target[date_col] = pd.to_datetime(self.target[date_col])
        self.target = self.target[(self.target[date_col] >= self.start) & (self.target[date_col]
           <= self.end)]

    def merge(self):

        dis = self.summary_t[[self.lat_col_target, self.lon_col_target]].values
        _, idx = self.kdtree.query(dis)

        site_to_src = {}
        for site, grid in zip(self.summary_t.index.tolist(), self.summary_g.index[idx]):
            site_to_src[site] = grid

        date_t = self.date_col_target
        date_s = self.date_col_source
        site_t = self.site_col_target
        range_index = self.date_index

        target_ = self.target.groupby(level=0).apply(lambda df: df.set_index
            (date_t).reindex(range_index))
        source_ = self.source.groupby(level=0).apply(lambda df: df.set_index
            (date_s).reindex(range_index))
        source_.index.names = ['_', '_']
        target_.index.names = [site_t, date_t] #only main frame name matters in merging!

        output = []
        for sitenumber, df_per_site in target_.groupby(level=0):
            df_per_site = df_per_site.reset_index(drop=True, level=0)
            df_per_site[site_t] = sitenumber
            additional_info = source_.loc[site_to_src[sitenumber], self.info_col_source].to_frame()
            output.append(pd.merge(df_per_site, additional_info, left_index=True, right_index=True))
        return pd.concat(output).reset_index()

if __name__ == '__main__':
    import pickle
    with open('./demo/grid_rain.pickle', 'rb') as handler:
        rain_grid = pickle.load(handler)
    with open('./demo/rain_flood.pickle', 'rb') as handler:
        flood = pickle.load(handler)['flood']

    merge = Merger('2015-01-01', '2015-12-31')
    merge.add_target(flood)
    merge.add_source(rain_grid)
    combined = merge.merge()
    print(combined.head())
