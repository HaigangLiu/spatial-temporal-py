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

def get_historical_median(dataframe, location_id='SITENUMBER', varname='GAGE_MAX'):
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
    if (location_id not in dataframe) or (varname not in dataframe):
        raise KeyError(f'make sure {location_id} and {variable_name} are in the dataframe')

    median_colname = '_'.join([varname, 'MEDIAN'])
    dev_colname = '_'.join([varname, 'DEV'])

    medians = dataframe.groupby(location_id)[varname].median().reset_index()
    dataframe = pd.merge(dataframe, medians, left_on=location_id, right_on=location_id, suffixes=['', '_MEDIAN'])
    dataframe[dev_colname] = dataframe[varname] - dataframe[median_colname]

    print(f'created the column {median_colname} to the dataframe')
    print(f'created the column {dev_colname} to show the change of {varname} relative to {median_colname}')
    print('-'*20)
    return dataframe

def apply_imputation(dataframe, spatial_column='SITENUMBER', temporal_column='DATE', varnames=['GAGE_MAX']):
    '''
    For all locations in the dataset, there is any missing value,
    we will impute the missing value by averaging the time before and the time point after.
    '''
    missing_before = np.sum(np.isnan(dataframe[varnames])) #test
    print(f'missing data status {missing_before}')

    def imputation(df):
        df.sort_values(temporal_column, axis=0)
        df.set_index(temporal_column, inplace=True)
        # df[varnames] =  df[varnames].interpolate(axis=0, method='time')
        dict_fill_na = df[varnames].mean(axis=0).to_dict()
        df[varnames] = df[varnames].fillna(dict_fill_na)
        return df

    dataframe = dataframe.groupby(spatial_column, group_keys=False).apply(imputation)
    missing_after = np.sum(np.isnan(dataframe[varnames]))
    print(f'missing data status after imputing {missing_after}')

    if missing_after.any():
        for varname in varnames:
            dataframe.loc[np.isnan(dataframe[varname]), varname] = np.mean(dataframe[varname])
        print('filled all other missing data with column mean')
    return dataframe.reset_index()  #test

def fill_missing_dates(df_original, spatial_column='SITENUMBER',
    temporal_column='DATE', method_to_fill_missing='nearest'):
    '''
    df_original (pandas dateframe): dataframe with missing dates
    spatial_column (string): the column name of stations
        since we assume there are multiple dates for the same locatino
    temporal_column (string): the name of column that contains dates
    '''
    start_counter = len(df_original)

    if (spatial_column not in df_original) or (temporal_column not in df_original):
        raise KeyError(f'make sure {spatial_column} and {temporal_column} are in the dataset')

    df = df_original.copy().reset_index(drop=True) #copy
    start_date = df[temporal_column].min()
    end_date = df[temporal_column].max()

    df = df_original.set_index([spatial_column, temporal_column]).unstack()
    df = df.reindex(pd.date_range(start_date, end_date),
           axis=1,
           level=1,
           fill_value=method_to_fill_missing).\
           stack().\
           reset_index()
    print(f'found {len(df) - start_counter} missing dates in the dataset.')
    return df

def remove_incomplete_locations(dataframe, station='SITENUMBER', varname=['GAGE_MAX'], threshold=0.90):
    '''
    Removes locations based on temporal completeness. Default is 90%, which means we discard
    locations with more than 10% of missing dates.
    We assume there is no missing dates for each location, if you suspect otherwise,
    use fill_missing_dates to preprocess data first.
    '''
    size_ = pd.Series(dataframe.groupby([station])[varname].size()) #no NA
    count_ = pd.Series(dataframe.groupby([station])[varname].count().values.ravel(), index=size_.index) #include na and everything

    start_counter = len(size_)
    ratio = (count_/size_).to_frame()
    ratio.columns = ['RATIO']

    df_with_ratio =pd.merge(dataframe, ratio, left_on=station, right_index=True)
    df_with_ratio = df_with_ratio[ df_with_ratio['RATIO'] >= threshold]
    df_with_ratio.drop(['RATIO'], axis=1, inplace=True)

    end_counter = len(pd.Series(df_with_ratio.groupby([station])[varname].size()))
    print(f'setting threshold to {threshold} has removed {start_counter - end_counter} locations')
    return df_with_ratio

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

def get_season_from_dates(dataframe, date_column='DATE', one_hot=True):
    '''
    Add a season indicator to the original data frame
    The added data frame will be called SEASON
    Additionally, if add_binary_for_each_season is True, then four more columns will be added.

    Args:
     Dataframe (pandas dataframe): original dataframe with date time information
     date_column (string): the name of the column that contains datetime information
     add_binary_for_each_season (boolean): if true, four more columns will be added: SPRING, FALL, SUMMER and WINTER.
    '''
    # dataframe[date_column] = pd.to_datetime(dataframe[date_column])

    def func(date):
        month = date.month
        if month in [1, 2, 12]:
            return 0
        elif month in [3,4,5]:
            return 1
        elif month in [6,7,8]:
            return 2
        else:
            return 3

    dataframe['SEASON'] = dataframe[date_column].map(func)
    if one_hot:
        dataframe = pd.get_dummies(dataframe, columns=['SEASON'])
        dataframe.rename({'SEASON_0': 'WINTER',
                          'SEASON_1': 'SPRING',
                          'SEASON_2': 'SUMMER',
                          'SEASON_3': 'FALL'})
    return dataframe

def transpose(df,
              spatial_col='SITENUMBER',
              temporal_col='DATE',
              start='2015-01-01',
              end='2015-12-31',
              constants = ['LATITUDE','LONGITUDE','ELEVATION', 'BASIN'],
              variables=['PRCP', 'GAGE_MAX_DEV']):

    df = df[(df.DATE <= pd.to_datetime(end)) & (df.DATE >= pd.to_datetime(start))]

    time_range = pd.date_range(start, end)
    x_ = variables
    product = [x_, time_range]

    colnames =[]
    for variable in variables:
        for day in range(len(time_range)):
            colnames.append(variable + '_' + str(day))

    transposed = df.set_index([spatial_col, temporal_col])[x_].unstack()
    transposed.reindex(pd.MultiIndex.from_product(product), axis=1)

    flat_and_wide = transposed.T.reset_index(level=0, drop=True).T
    flat_and_wide.columns = colnames

    constants.append(spatial_col)
    constants_df = df[constants]
    constants_df = constants_df.groupby(spatial_col).first()

    output = pd.merge(constants_df, flat_and_wide,  left_index=True, right_index=True).reset_index()
    return output

def mark_flood_season(original_df, start='2015-10-01', end='2015-12-31', time_col='DATE', name='FLOOD_SEASON'):
    '''
    add a dummy variable for flood season.
    This is because the rainfall and flood dynamics might be different on normal days vs. flood seasons.
    use argwhere can be 6 times faster than using a loop.
    '''
    criteria = (original_df[time_col] >= start) & (original_df[time_col] <= end)
    original_df[name] = np.where(criteria, 1, 0)
    return original_df

def get_autoregressive_terms(df, steps=1, groupby_var='SITENUMBER',
    date_var='DATE', variable='GAGE_MAX'):
    '''
    simple add np.nan in the first position, and shift everyone back a slot
    This is complicated by the existence of multiple stations.
    Hence, we have to group by sitenumber, and for each set do the aforementioned operations.

    dataframe(pandas dataframe): a spatial temporal dataframe
    steps(int): the number of autoreg term to be generated (default is 1)
    groupby(str): the location/site identifier column. Each station/site should have its own time series default value (groupby)
    variable(str): the name of the target column; default value 'DEV_GAGE_MAX'
    '''
    if date_var not in df:
        raise KeyError(f'cannot find {date_var}. Need to sort values by dates')
        return None

    if groupby_var not in df:
        raise KeyError(f'cannot find {groupby_var}. Need to sort dates by station')
        return None

    dataframe = df.copy()
    dataframe.sort_values([groupby_var, date_var])

    for i in range(steps):
        additional_rows = dataframe.groupby(groupby_var)[variable].shift(i+1).to_frame()
        dataframe = pd.merge(dataframe, additional_rows,
                      left_index=True,
                      right_index=True,
                      suffixes=['', '_MINUS_'+ str(i+1)])
    return dataframe

if __name__ == '__main__':
    from datetime import datetime
    parse_dates= ['DATE']
    f = pd.read_csv('./data/flood_data_daily_beta.csv', index_col=0, dtype={'SITENUMBER': str}, parse_dates=parse_dates)
    f = f[(f.DATE >= datetime(2010, 1, 1)) & (f.DATE <= datetime(2016, 12, 31))]

    f = fill_missing_dates(f)
    f = get_season_from_dates(f)
    f = remove_incomplete_locations(f)
    f = mark_flood_season(f)
    f = get_historical_median(f)
    f = get_autoregressive_terms(f)

    f = apply_imputation(f)
    f = get_elevation(f) #0.7
    f = get_watershed(f) #0.5
    f = get_basin_from_watershed(f) #0.1

    f = transpose(f, variables=['GAGE_MAX_DEV'])
    print(f.shape)

    # from merge_rain_and_flood import Merger
    # print('start merging')
    # raw_df = Merger(raw_df, rain_df, '2010-01-01', '2016-12-31').run()
    # print('finished merging')
    # raw_df.drop(['index'], axis=1, inplace=True)

