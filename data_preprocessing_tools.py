import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from utility_functions import get_in_between_dates
WATERSHED_PATH ='./data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'

def get_elevation(input, key=None, lat=None, lon=None):
    '''
    find the elevation for given latitude and longitude
    the input can be of list like [lon, lat]
    or a dataframe that contains longitude column and latitude column

    Args:
        input (pandas dataframe, or list)
        key: if dataframe is given, key must be provided to indicate which
        column is station name
    Return:
        if input is a list:
            returns the height as a float number
        if input is dataframe:
            returns the original dataframe within an additional column for
            elevation
    '''
    source_1 = './data/shape_file/elevation_info/srtm_20_05/srtm_20_05.tif'
    source_2 = './data/shape_file/elevation_info/srtm_20_06/srtm_20_06.tif'
    source_3 = './data/shape_file/elevation_info/srtm_21_05/srtm_21_05.tif'
    source_4 = './data/shape_file/elevation_info/srtm_21_06/srtm_21_06.tif'
    source_collection = [source_1, source_2, source_3, source_4]

    def get_elevation_one_loc(input):
        elevation = []
        for source in source_collection:
            with rasterio.open(source) as f:
                try:
                    vals = f.sample([input])
                    for val in vals:
                        # if val[0] >= 0: #lowest point of sc is sea level
                        elevation.append(val[0])
                        break
                except IndexError:
                    continue

        return max(elevation)

    if isinstance(input, list):
        print('assuming the format is [longitude, latitude]')
        print('the unit is measured in meter')
        return get_elevation_one_loc(input)

    elif isinstance(input, pd.DataFrame):
        input_df = input.copy()

        lat = 'LATITUDE' if lat is None else lat
        lon = 'LONGITUDE' if lon is None else lon
        key = 'SITENUMBER' if key is None else key

        try:
            #construct a summary table to enhance performance
            summary = input_df.groupby([key]).first().reset_index()[[key, lon, lat]]

            input_df.set_index(key, inplace=True)
            for idx, row in summary.iterrows():
                key, lat, lon = row.values
                height = get_elevation_one_loc([lat, lon])
                input_df.loc[key, 'ELEVATION'] = height

        except KeyError:
            print(f'cannot find one or more of the following column names: {lat}, {lon} and {key}')
            print('please double check the column name')
            return None
    print('created a new column called ELEVATION to store the height information')
    return input_df.reset_index()

def get_watershed(input, shapfile=None, key=None, lat=None, lon=None):
    '''
    for each location, find which watershed it belongs to.
    A attribute 'data' will be generated which is a new
    dataframe with an additional column called WATERSHED
    '''
    if shapfile is None:
        huc8_units = gpd.read_file(WATERSHED_PATH)
        water_shed_info = huc8_units[['NAME', 'geometry']]
        # water_shed_info = prep(water_shed_info)

    def get_watershed_for_one_loc(list_):
        for _, rows in water_shed_info.iterrows():
                    name, polygon = rows
                    if Point(list_).within(polygon):
                        return name
        else:
            raise ValueError('watershed not found')

    if isinstance(input, list):
        print('assuming the format is [longitude, latitude]')
        print('the unit is measured in meter')
        return get_watershed_for_one_loc(input)

    elif isinstance(input, pd.DataFrame):
        input_df = input.copy()

        lat = 'LATITUDE' if lat is None else lat
        lon = 'LONGITUDE' if lon is None else lon
        key = 'SITENUMBER' if key is None else key

        try:
            summary = input_df.groupby([key]).first().reset_index()[[key, lon, lat]] #better performance
        except KeyError:
            print(f'cannot find one or more of the following colums: {lat}, {lon} and {key}')
            print('please doublec check the column name.')
            return None

        input_df.set_index(key, inplace=True)
        for idx, row in summary.iterrows():
            key_, lat, lon = row.values
            watershed_name = get_watershed_for_one_loc([lat, lon])
            input_df.loc[key_, 'WATERSHED'] = watershed_name

        input_df.reset_index(inplace=True)

        if True:
            number_of_obs = input_df.groupby(['WATERSHED']).count()[key]
            singular_huc_areas = number_of_obs[number_of_obs<=1].index
            input_df = input_df[~input_df.WATERSHED.isin(singular_huc_areas)]
        print('created a new column called WATERSHED to store the huc watershed information')
        return input_df.reset_index()

def get_historical_median(dataframe, location_id=None, varname=None, delete_empty_loc=True):
    '''
    For spatial temporal data, find the median for each locations.
    Args:
        dataframe (pandas dataframe)
        location_id (string): the column in the dataframe that contains location names
        varname (string): the column name of the variable in the dataframe
        delete_empty_loc (boolean): drop this location if the median is 0
            note: this is because median is nan iff there is no available data at all.
    Return:
        the dataframe with an added column called historical median
    '''
    varname = 'GAGE_MAX' if varname is None else varname
    location_id = 'SITENUMBER' if location_id is None else location_id
    new_name = '_'.join(['HISTORICAL_MEDIAN', varname])

    try:
        location_list = dataframe[location_id].unique().tolist()
        dataframe = dataframe.set_index(location_id) #it's copy. leave original
        # intact
        for location in location_list:
            series_ = pd.Series(dataframe.loc[location, varname])
            if len(series_) == 1:
                raise TypeError('Only get one value for the given time and location')
            dataframe.loc[location, new_name] = series_.median()
            if delete_empty_loc:
                if np.isnan(series_.median()):
                    print(f'location {location} is dropped because the median is NA')
                    dataframe.drop(dataframe.loc[location].index, axis=0, inplace=True)
        print(f'added the column {new_name} to the dataframe')

    except KeyError:
        print('An error occurred: make sure the column names of variable and location are spelled correctly.')
        return None
    return dataframe.reset_index(drop=False)

def apply_imputation(dataframe, spatial_column=None, temporal_column=None,varname=None):
    '''
    For all locations in the dataset, there is any missing value,
    we will impute the missing value by averaging the time before
    and the time point after.
    '''
    spatial_column = 'SITENUMBER' if spatial_column is None else spatial_column
    temporal_column = 'DATE' if temporal_column is None else temporal_column
    varname = 'GAGE_MAX' if varname is None else varname

    dataframe = dataframe.sort_values(['SITENUMBER', 'DATE']) #copy
    begin = sum(np.isnan(dataframe[varname])) #test

    try:
        location_list = dataframe[spatial_column].unique().tolist()
        dataframe.set_index(spatial_column, inplace=True)
        for location in location_list:
            dataframe.loc[location, varname] = dataframe.loc[location, varname].interpolate()
    except KeyError:
        print('make sure the spelling of the name of spatial-related column and temporal related column is right.')
        print('spatial-related column usually comes as station, or station id. Default value is SITENUMBER')
        print('temporal-related column usually comes as date or year, default value is DATE.')
    end = sum(np.isnan(dataframe[varname])) #test
    print(begin, end)
    return dataframe.reset_index(drop=False)


def fill_missing_dates(df_original, spatial_column=None, temporal_column=None, fixed_vars=None):
    '''
    df_original (pandas dateframe): dataframe with missing dates
    spatial_column (string): the column name of stations
        since we assume there are multiple dates for the same locatino
    temporal_column (string): the name of column that contains dates
    fixed_vars (list): specify the variables that does not change with time. e.g. (latitude, longitude, elevations etc.)
        Not specifying this may cause severe missing data issues
    '''

    spatial_column = 'SITENUMBER' if spatial_column is None else spatial_column
    temporal_column = 'DATE' if temporal_column is None else temporal_column

    df_original = df_original.reset_index(drop=True) #copy
    unique_days = get_in_between_dates(df_original[temporal_column].min(), df_original[temporal_column].max())
    print(df_original[spatial_column])
    unique_locs = pd.unique(df_original[spatial_column])

    number_of_locs = len(unique_locs)
    number_of_days = len(unique_days)

    df_new = pd.DataFrame(columns=['SITENUMBER_x', 'DATE_x'])
    df_new['SITENUMBER_x'] = np.repeat(unique_locs, number_of_days)
    df_new['DATE_x'] = np.tile(unique_days, number_of_locs)

    if fixed_vars is not None:
        summary_file = df_original.groupby(spatial_column).first().reset_index() #copy
        df_new.set_index('SITENUMBER_x', inplace=True)
        for idx, row in summary_file.iterrows():
            values = row[fixed_vars]
            identifier = row[spatial_column]
            for var_to_propagate in fixed_vars:
                df_new.loc[identifier, var_to_propagate] = values[var_to_propagate]
        df_new.reset_index(inplace=True)

    output = pd.merge(df_new, df_original, how='left', left_on=['SITENUMBER_x', 'DATE_x'], right_on=[spatial_column, temporal_column])

    drop_list = [column for column in output.columns if column.endswith('_y')]#clean up
    drop_list.extend([temporal_column, spatial_column])
    output.drop(drop_list, inplace=True, axis=1)
    output.columns = [column.replace('_x', '') for column in output.columns]
    return output

if __name__ == '__main__':
    from SampleDataLoader import load_rainfall_data
    # rainfall = load_rainfall_data('monthly')

    file_ = pd.read_csv('./data/rainfall_and_flood_10_beta.csv', index_col=0, dtype={'SITENUMBER':str})

    median_added = get_historical_median(file_, varname='GAGE_MAX', location_id= 'SITENUMBER', delete_empty_loc=False)
    simple_imputatation(median_added, spatial_column='SITENUMBER', varname='GAGE_MAX')
    print(file_.columns)
