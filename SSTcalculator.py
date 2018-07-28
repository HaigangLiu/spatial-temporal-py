import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from datetime import datetime
from dateutil.relativedelta import relativedelta
from utilities_functions import coordinates_converter
class SSTcalculator:

    def __init__(self, nc_data_dir, nc_mask_dir):

        '''
        Args:
            nc_data_dir (str): The directory to the sea surface temperature
            nc_mask_dir (str): The directory to mask the land from sea.
        '''

        self.nc_data =Dataset(nc_data_dir, mode = 'r')
        self.nc_mask = Dataset(nc_mask_dir, mode = 'r')
        self.nc_mask = np.array(self.nc_mask['mask'][:].data, dtype = np.bool)
        self.lats = self.nc_data['lat'][:]
        self.lons = self.nc_data['lon'][:]
        self.sst = self.nc_data['sst'][:]

        self.lons = np.array([term if term <= 180 else (term -360) for term in self.lons])


    def _month_locator(self, starting_date, end_date):
        time_values = self.nc_data.variables['time'][:]
        sensible_dates_values = pd.to_datetime('1800-01-01') + pd.to_timedelta(time_values , unit = "d")
        time_dict = {str(sensible_dates_values[i])[0:7]: i for i in np.arange(len(time_values))}
        idx = np.arange(time_dict[starting_date], time_dict[end_date] + 1)
        return idx

    def _monthly_look_up(self, month_idx, spatial_points, n_neighbors = 5):

        sst = self.sst[:]
        sst.data[month_idx,:,:][~self.nc_mask[0,:,:]] = np.nan #noland！should be around 70% percent of original data

        sst_df_sea_only = pd.DataFrame(sst.data[month_idx,:,:])
        sst_df_sea_only.index = self.lats
        sst_df_sea_only.columns = self.lons

        sst_df_sea_only_ = sst_df_sea_only.stack().reset_index()
        sst_df_sea_only_.columns = ['LATITUDE', 'LONGITUDE', 'SST']

        cartesian_coord_trees = coordinates_converter(sst_df_sea_only_)
        tree = cKDTree(cartesian_coord_trees.values)
        cartesian_coord_query = coordinates_converter(spatial_points)
        distances, idx = tree.query(cartesian_coord_query.values, k = n_neighbors)

        sst_list = []
        for i in range(len(spatial_points)):
            average_sst = sst_df_sea_only_.iloc[idx[i]].mean().SST
            sst_list.append(average_sst)

        spatial_points['SST'] = sst_list
        return spatial_points

    def look_up_engine(self, starting_date, end_date, spatial_points):

        start_date_ = pd.to_datetime(starting_date)
        current_month = start_date_.month
        current_year = start_date_.year
        snapshots = []
        indices =  self._month_locator(starting_date, end_date)

        for index in indices:
            spatial_points = self._monthly_look_up(index, spatial_points)

            spatial_points_cp = spatial_points.copy()
            spatial_points_cp.loc[:,'MONTH'] = current_month
            spatial_points_cp.loc[:,'YEAR'] = current_year
            snapshots.append(spatial_points_cp)

            start_date_ = start_date_ + relativedelta(months=1)
            current_month = start_date_.month
            current_year = start_date_.year

        return snapshots

if __name__ == '__main__':

    nc_mask_dir = '/Users/haigangliu/Dropbox/DataRepository/data_file_new/lsmask.nc'
    nc_data_dir = '/Users/haigangliu/Dropbox/DissertationCode/sst/sst.mnmean.nc'

    new_df = pd.read_csv('./data/monthly_rainfall.csv')

    locs = new_df.groupby('STATION').first().reset_index()[['STATION','LONGITUDE', 'LATITUDE', 'ELEVATION']]
    lats_and_lons = locs[['LONGITUDE', 'LATITUDE']]

    test_case = SSTcalculator(nc_data_dir, nc_mask_dir)
    s = test_case.look_up_engine('2011-10', '2012-12', lats_and_lons)
