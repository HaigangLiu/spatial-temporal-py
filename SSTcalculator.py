import pandas as pd
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
from utility_functions import coordinates_converter

class SSTcalculator:

    def __init__(self, locations, start_year, start_month, end_year, end_month, n_neighbors=5, nc_data_dir=None, nc_mask_dir=None):
        '''
        Args:
            nc_data_dir (str): The directory to the sea surface temperature
            nc_mask_dir (str): The directory to mask the land from sea.
        '''
        if nc_data_dir is None:
            nc_mask_dir = './data/lsmask.nc'
        if nc_data_dir is None:
            nc_data_dir = './data/sst.mnmean.nc'

        self.nc_data = Dataset(nc_data_dir, mode = 'r')
        self.nc_mask = Dataset(nc_mask_dir, mode = 'r')
        self.nc_mask = np.array(self.nc_mask['mask'][0].data, dtype = np.bool)

        self.lats = self.nc_data.variables['lat'][:]
        self.lons = self.nc_data.variables['lon'][:]
        self.sst = self.nc_data.variables['sst'][:]
        self.time = self.nc_data.variables['time'][:]
        self.lons = np.array([term if term <= 180 else (term -360) for term in self.lons])

        self.start_year = int(start_year)
        self.start_month = int(start_month)

        self.start_date = date(self.start_year, self.start_month, int(1))
        self.end_date = date(int(end_year), int(end_month), int(1))
        self.n_neighbors = int(n_neighbors)
        self.spatial_points = locations

    def _find_time_index(self):

        converted_dates = {date(1800, 1,1) + timedelta(int(t)): i for i, t in enumerate(self.time)} #start with 1800-01-01
        start_date_idx = converted_dates.get(self.start_date, None)
        end_date_idx = converted_dates.get(self.end_date, None)

        if start_date_idx and end_date_idx:
            return range(start_date_idx, end_date_idx+1)
        else:
            raise ValueError('the database has only monthly data from 1981 to 2017')

    def _monthly_look_up(self, month_idx):

        #noland！should be around 70% percent of original data
        base = self.spatial_points.copy()
        sst_df_sea_only = pd.DataFrame(self.sst.data[month_idx,:,:])
        sst_df_sea_only.index = self.lats
        sst_df_sea_only.columns = self.lons

        sst_df_sea_only_ = sst_df_sea_only.stack().reset_index()
        sst_df_sea_only_.columns = ['LATITUDE', 'LONGITUDE', 'SST']

        cartesian_coord_trees = coordinates_converter(sst_df_sea_only_)
        tree = cKDTree(cartesian_coord_trees.values)
        cartesian_coord_query = coordinates_converter(self.spatial_points)
        distances, idx = tree.query(cartesian_coord_query.values, k=self.n_neighbors)

        sst_list = []
        for i in range(len(self.spatial_points)):
            sst_list.append(sst_df_sea_only_.iloc[idx[i]].mean().SST)

        base['SST'] = sst_list
        return base

    def run(self):

        snapshots = []
        indices =  self._find_time_index()

        start_year_  = self.start_year
        start_month_ = self.start_month
        start_date_ = self.start_date

        for index in indices:
            self.sst.data[index,:,:][~self.nc_mask] = np.nan
            df_single_month = self._monthly_look_up(index)

            df_single_month.loc[:,'MONTH'] = start_month_
            df_single_month.loc[:,'YEAR'] = start_year_
            snapshots.append(df_single_month)

            start_date_ = start_date_ + relativedelta(months=1)
            current_month = start_date_.month
            current_year = start_date_.year

        return pd.concat(snapshots, ignore_index=True)

if __name__ == '__main__':

    from SampleDataLoader import load_rainfall_data
    new_df = load_rainfall_data()
    locs = new_df.groupby('STATION').first().reset_index()[['STATION','LONGITUDE', 'LATITUDE']]
    lats_and_lons = locs[['LONGITUDE', 'LATITUDE']]
    test_case = SSTcalculator(lats_and_lons, 2011, 10, 2011, 11).run()
    print(test_case.SST.describe())
