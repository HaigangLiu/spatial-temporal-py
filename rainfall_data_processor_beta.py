import pandas as pd
from netCDF4 import Dataset
import numpy.ma as ma
from pyproj import Proj, transform
from tqdm import tqdm
import numpy as np

def unpack_nc_file(file_dir, coordinate_names, variable, to_lat_lon=True):
    '''
    A parser function for netCDF4 file.
    file_dir (string): the directory to netCDF4 file
    to_lat_lon (boolean): if true, this method will convert location in
        latitude and longitude automatically
    variable (string): the name of the variable of interest
    '''
    netcdf4_dataset = Dataset(file_dir)
    coordinate_a,  coordinate_b = coordinate_names

    x_axis = ma.getdata(netcdf4_dataset[coordinate_a][:])#1121,first
    y_axis = ma.getdata(netcdf4_dataset[coordinate_b][:])#881

    obs = ma.getdata(netcdf4_dataset[variable][:])
    dataframe = pd.DataFrame(obs)
    dataframe.index = y_axis; dataframe.columns = x_axis

    dataframe_ = dataframe.stack().reset_index()
    dataframe_.columns = [coordinate_a, coordinate_b, variable]

    if to_lat_lon:
        try:
            original_projection_scheme = netcdf4_dataset['crs'].proj4
        except KeyError:
            print('this netCDF4 file does not contain crs info')
            return None

        target_projection_scheme = 'epsg:4326' #longitude and latitude
        in_projection = Proj(original_projection_scheme)
        out_projection = Proj(init=target_projection_scheme)
        longitude = []; latitude = []

        for idx, row in tqdm(dataframe_[[coordinate_a,coordinate_b]].iterrows()):
            y1, x1 = row.values
            y2, x2 = transform(in_projection,out_projection,x1,y1)
            longitude.append(y2); latitude.append(x2)

        dataframe_['LONGITUDE'] = longitude
        dataframe_['LATITUDE'] = latitude

    return dataframe_

def nc_dataset_postprocess(dataframe):

    data_frame_copy = data_frame.copy()
    data_frame_copy.PRCP[data_frame_copy.PRCP<0] = np.nan
    missing_data_removed = data_frame_copy[data_frame_copy.PRCP>0]
    return missing_data_removed

if __name__ == '__main__':

    file_dir = './data/rainfall_last180.nc'
    df = unpack_nc_file(file_dir, ['x', 'y'], variable='observation',
        to_lat_lon= True)
    df_2 = unpack_nc_file(file_dir, ['x', 'y'], variable='observation',
        to_lat_lon = False)
    print(df.head())
    print(df_2.head())
