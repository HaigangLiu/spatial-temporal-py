import pandas as pd
import numpy as np
import fiona
import rasterio
from shapely.geometry import Point,shape
from shapely.prepared import prep
from utility_functions import get_in_between_dates, get_dict_basins_to_watershed

WATERSHED_PATH = './data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'

def get_elevation(input, key=None, lat=None, lon=None):
    '''
    find the elevation for given latitude and longitude
    the input can be of list like [lon, lat]
    or a dataframe that contains longitude column and latitude column
    Args:
        input (pandas dataframe, or list)
        key: if dataframe is given, key must be provided to indicate which column is station name
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
        return max(0, max(elevation)) #lowest point in sc is 0

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
    print('-'*20)
    return input_df.reset_index()

def get_watershed(dataframe, shapefile=None, location_id=None, lat=None, lon=None, singluar_removal=False):
    '''
    add a column with watershed information
    Args:
        dataframe: pandas dataframe
        shapefile (string, dir to an shp file): shape information of watersheds
        location_id (string): the name of column of location names/ids
        lat (string): the name of the column of latitude information
        lon (string): the name of the column of longitude information
        singular_removal: boolean if true the isolated locations wille be removed.
    '''
    shapefile = WATERSHED_PATH if shapefile is None else shapefile
    location_id = 'SITENUMBER' if location_id is None else location_id
    lat = 'LATITUDE' if lat is None else lat
    lon = 'LONGITUDE' if lon is None else lon

    lats_and_lons = dataframe.groupby(location_id).first()[[lat,lon]].reset_index()

    shapes = []; names = []
    for i in fiona.open(shapefile):
        names.append(i['properties']['NAME'])
        shapes.append(prep(shape(i['geometry'])))

    dict_ = {}
    for idx, row in lats_and_lons.iterrows():
        sitenumber, lat, lon = row.values
        point = Point([ lon, lat])
        for watershed_name, shape_ in zip(names, shapes):
            if shape_.contains(point):
                dict_[sitenumber] = watershed_name
                break

    dataframe['WATERSHED'] = dataframe[location_id].map(dict_)
    if singluar_removal:
        number_of_obs = dataframe.groupby(['WATERSHED']).count()[key]
        singular_huc_areas = number_of_obs[number_of_obs<=1].index
        dataframe = dataframe[~dataframe.WATERSHED.isin(singular_huc_areas)]

    return dataframe

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
    dev_name = '_'.join(['DEV', varname])
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
    except KeyError:
        print('An error occurred: make sure the column names of variable and location are spelled correctly.')
        return None
    dataframe[dev_name] = dataframe[varname] - dataframe[new_name]
    print(f'created the column {new_name} to the dataframe')
    print(f'created the column {dev_name} to show the change of {varname} relative to {new_name}')
    print('-'*20)
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

    print('the imputation is done sucessfully.')
    print(f'there are {begin} missing values in {varname} before and {end} missing values after')
    print('-'*20)
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
    start_date = df_original[temporal_column].min()
    end_date = df_original[temporal_column].max()
    unique_days = get_in_between_dates(start_date, end_date)

    print('a new record will be added for any missing dates in the file')
    print(f'starts from {start_date}, ends with {end_date}')
    print(f'corresponding variables with be filled with NA')
    print('-'*20)

    unique_locs = pd.unique(df_original[spatial_column])
    number_of_locs = len(unique_locs)
    number_of_days = len(unique_days)

    df_new = pd.DataFrame(columns=['SITENUMBER_x', 'DATE_x'])
    df_new['SITENUMBER_x'] = np.repeat(unique_locs, number_of_days)
    df_new['DATE_x'] = np.tile(unique_days, number_of_locs)

    summary_file = df_original.groupby(spatial_column).first().reset_index() #copy
    keys = summary_file[spatial_column].tolist()

    if fixed_vars:
        for fixed_var in fixed_vars: #write a dict and then lookup
            lookup_t = {k:v for k, v in zip(keys, summary_file[fixed_var].tolist())}
            df_new[fixed_var] = df_new['SITENUMBER_x'].map(lookup_t)

    output = pd.merge(df_new, df_original, how='left', left_on=['SITENUMBER_x', 'DATE_x'], right_on=[spatial_column, temporal_column])
    drop_list = [column for column in output.columns if column.endswith('_y')]#clean up
    drop_list.extend([temporal_column, spatial_column])
    output.drop(drop_list, inplace=True, axis=1)
    output.columns = [column.replace('_x', '') for column in output.columns]
    return output

def filter_missing_locations(dataframe, varname=None, years_from_now=7, threshold=0.9, spatial_column=None, temporal_column=None, last_year=2016):

    varname = 'GAGE_MAX' if varname is None else varname
    spatial_column = 'SITENUMBER' if spatial_column is None else spatial_column
    temporal_column = 'DATE' if temporal_column is None else temporal_column
    begin = len(pd.unique(dataframe[spatial_column]).tolist())

    df_copy = dataframe.copy()
    df_copy[temporal_column] = pd.to_datetime(df_copy[temporal_column]) #set value
    df_copy = df_copy.set_index([temporal_column, spatial_column]).unstack()
    yearly_non_missing_count = df_copy.resample('Y').count() #will count non-na automatically

    number_of_days = []; index_ =[]
    for i in yearly_non_missing_count.index: #for leap years
        if i.year%4 == 0:
            number_of_days.append(366)
            index_.append(str(i.year))
        else:
            number_of_days.append(365)
            index_.append(str(i.year))

    yearly_non_missing_count.index = index_
    report = (yearly_non_missing_count[varname].T/np.array(number_of_days)).reset_index()

    #keep list
    years = [str(last_year-year) for year in range(years_from_now)]
    report.set_index(spatial_column, inplace=True)

    data_in_range = report[years]

    keeper = data_in_range[data_in_range > threshold]
    keeper.dropna(axis=0, inplace=True)
    station_list = keeper.index.tolist()

    mask = dataframe[spatial_column].isin(station_list)
    end = len(station_list)

    print(f'stations with data completeness less than {threshold} are discarded.')
    print(f'out of {begin} stations, {end} stations are kept.')
    print('-'*20)
    return dataframe[mask]

def get_basin_from_watershed(dataframe, watershed_col_name=None):

    new_source = './basin_list_updated.txt'
    dict_ = get_dict_basins_to_watershed(new_source, mode='name', reverse=True)

    basin = 'BASIN'
    watershed_col_name = 'WATERSHED' if watershed_col_name is None else watershed_col_name
    try:
        dataframe[basin] = dataframe[watershed_col_name].map(dict_)
    except KeyError:
        print(f'{watershed_col_name} does not exist. Check the spelling of column name.')
        return None

    print(f'created the basin information in a column called {basin}')
    print('-'*20)
    return dataframe

def get_season_from_dates(dataframe, date_column=None, add_binary_for_each_season=True):
    '''
    Add a season indicator to the original data frame
    The added data frame will be called SEASON
    Additionally, if add_binary_for_each_season is True, then four more columns will be added.

    Args:
     Dataframe (pandas dataframe): original dataframe with date time information
     date_column (string): the name of the column that contains datetime information
     add_binary_for_each_season (boolean): if true, four more columns will be added: SPRING, FALL, SUMMER and WINTER.
    '''
    date_column = 'DATE' if date_column is None else date_column
    dates = dataframe[date_column].values.tolist()

    seasons = []
    for date in dates:
        month = date.split('-')[1]
        if month in ['01', '02','12']:
            seasons.append('WINTER')
        elif month in ['03', '04','05']:
            seasons.append('SPRING')
        elif month in ['06', '07', '08']:
            seasons.append('SUMMER')
        elif month in ['09', '10', '11']:
            seasons.append('FALL')
        else:
            raise ValueError('the month is not readable')
    dataframe['SEASON'] = np.array(seasons)
    if add_binary_for_each_season:
        four_season = pd.get_dummies(np.array(seasons))
        dataframe = pd.concat([dataframe, four_season], axis=1)
    return dataframe

def transpose_dataframe(original_df,
                   temporal_col='DATE',
                   start=None,
                   end=None,
                   key='SITENUMBER',
                   fixed_variables = ['LATITUDE','LONGITUDE','ELEVATION', 'BASIN'],
                   time_varying_variables=['PRCP', 'DEV_GAGE_MAX']):
    '''
    Generate a flat-and-wide data frame based on a deep one.
    In short, the original dataframe has a date column and a variable e.g. rainfall amount
    the returned data frame will have a rainfall amount column for each day.
    '''
    start = min(original_df[temporal_col]) if start is None else start
    end = max(original_df[temporal_col]) if end is None else end

    dates_list = get_in_between_dates(start, end)
    selected_data = original_df[original_df[temporal_col].isin(dates_list)]

    summary_form = selected_data.groupby(key).first()[fixed_variables].reset_index()

    temporal_variable = [key, temporal_col]
    temporal_variable.extend(time_varying_variables)

    temp = selected_data[temporal_variable]
    output_df = temp.set_index([key, temporal_col]).unstack()

    new_columns = []
    for variable_name in time_varying_variables:
        for month in range(len(dates_list)):
            new_var_name = '_'.join([variable_name, str(month + 1)])
            new_columns.append(new_var_name)

    output_df.columns = new_columns
    output_df.reset_index(inplace=True)

    dataframe = summary_form.merge(output_df, left_on=key, right_on=key, how='inner')
    return dataframe

def mark_flood_season(original_df, start, end, time_col='DATE'):
    '''
    add a dummy variable for flood season.
    This is because the rainfall and flood dynamics might be different on normal days vs. flood seasons.
    '''
    flood_season_dates = get_in_between_dates(start, end)
    indicators = []
    for date_ in original_df[time_col].isin(list_of_dates):
        if date_:
            booleans.append(1)
        else:
            booleans.append(0)
    original_df['FLOOD_SEASON'] = np.array(indicators)
    return original_df

if __name__ == '__main__':

    # raw_df = pd.read_csv('./data/rainfall_and_flood_10_beta.csv', index_col=0, dtype={'SITENUMBER': str})

    checkout_df = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    ss = transpose_dataframe(checkout_df, start='2010-01-01', end='2010-01-02')

    print(get_elevation([ -80.03, 33]))
    rain_df = pd.read_csv('./demo/SC_20050101-20170627-19b7.txt', delimiter=" ")
    raw_df = pd.read_csv('./data/flood_data_daily_beta.csv', index_col=0, dtype={'SITENUMBER': str})
    raw_df = fill_missing_dates(raw_df, fixed_vars = ['LATITUDE','LONGITUDE']) #0.7
    from merge_rain_and_flood import Merger

    print('start merging')
    raw_df = Merger(raw_df, rain_df, '2010-01-01', '2016-12-31').run()
    print('finished merging')
    # raw_df.drop(['index'], axis=1, inplace=True)
    # should fill missing dates first
    # then merge...

    raw_df = filter_missing_locations(raw_df, varname='GAGE_MAX', threshold=0.85) #0.5
    raw_df = apply_imputation(raw_df, varname='GAGE_MAX') #0.4
    #explain this part #e
    raw_df = get_historical_median(raw_df, varname='GAGE_MAX') #0.3
    raw_df = get_elevation(raw_df) #0.7
    raw_df = get_watershed(raw_df) #0.5
    raw_df = get_basin_from_watershed(raw_df) #0.1
    raw_df.to_csv('./data/check_out.csv') #13.6 -> 6.3

