import os
import pandas as pd
import numpy as np
import fiona
import rasterio
from shapely.geometry import Point,shape
from shapely.prepared import prep
from utility_functions import get_in_between_dates, get_dict_basins_to_watershed

WATERSHED_PATH = './data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'

def get_median_by_site(dataframe, varname='GAGE_MAX', site='SITENUMBER'):
    '''
    For spatial temporal data, find the median for each locations.
    Args:
        dataframe (pandas dataframe)
        site (string): the column in the dataframe that contains location names
        varname (string): the column name of the variable in the dataframe
    Return:
        the dataframe with a historical median and a residual column
    '''
    median_colname = '_'.join([varname, 'MEDIAN'])
    dev_colname = '_'.join([varname, 'DEV'])

    try:
        medians = dataframe.groupby(site)[varname].median().reset_index()
    except KeyError:
        print(f'make use both {varname} and {site} are in the dataframe.')
        return None

    dataframe = pd.merge(dataframe, medians, left_on=site, right_on=site, suffixes=['', '_MEDIAN'])
    dataframe[dev_colname] = dataframe[varname] - dataframe[median_colname]

    print(f'added the column {median_colname} to the dataframe')
    print(f'added the column {dev_colname} to show the deviation from median')
    print('-'*20)
    return dataframe

def transpose_dataframe(original_df,
                   date_col='DATE',
                   key_col='SITENUMBER',
                   fixed_var=['LATITUDE','LONGITUDE','ELEVATION', 'BASIN'],
                   remove_vars=None):
    '''
    Generate a flat-and-wide data frame based on a deep one.
    In short, the original dataframe has a date column and a variable e.g. rainfall amount
    the returned data frame will have a rainfall amount column for each day.
    '''
    all_variables = original_df.columns.tolist()
    if remove_vars:
        selected_data = original_df[[col for col in all_variables if col not
        in remove_vars]]
    else:
        selected_data = original_df

    other_variables = [date_col, key_col]
    if fixed_var:
        other_variables.extend(fixed_var)
    for col in other_variables:
        try:
            all_variables.remove(col)
        except ValueError:
            print(f'{col} does not exist in the given dataframe')
            return None

    temporal_variable = [key_col, date_col]
    temporal_variable.extend(all_variables)

    temp = selected_data[temporal_variable]
    output_df = temp.set_index([key_col, date_col]).unstack()

    col_names = []
    dates_raw = original_df[date_col].unique()

    for variable_name in all_variables:
        for month in range(len(dates_raw)):
            new_var_name = '_'.join([variable_name, str(month + 1)])
            col_names.append(new_var_name)

    output_df.columns = col_names
    output_df.reset_index(inplace=True)

    if fixed_var: #handle info like latitude and longitude
        summary_form = selected_data.groupby(key_col).first()[fixed_var].reset_index()
        output_df = summary_form.merge(output_df, left_on=key_col, right_on=key_col, how='inner')

    return output_df

def get_season_from_dates(dataframe, date_column='DATE', one_hot=True, copy=True):
    '''
    Add a season indicator to the original data frame
    The added data frame will be called SEASON
    Additionally, if add_binary_for_each_season is True, then four more columns will be added.

    Args:
     Dataframe (pandas dataframe): original dataframe with date time information
     date_column (string): the name of the column that contains datetime information
     one_hot (binary): if true, four columns (instead of one) will be added: SPRING, FALL, SUMMER and WINTER.
                       if false, then just one column SEASON will be added.
    '''
    if copy:
        dataframe = dataframe.copy()

    dict_month_to_season = {3: 'SPRING', 4: 'SPRING', 5: 'SPRING',
                            6: 'SUMMER', 7: 'SUMMER', 8: 'SUMMER',
                            9: 'FALL', 10: 'FALL', 11: 'FALL',
                            12: 'WINTER', 1: 'WINTER', 2: 'WINTER'}
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    dataframe['SEASON'] = dataframe['DATE'].dt.month.map(dict_month_to_season)

    if one_hot:
        seasons = ['WINTER', 'SPRING', 'SUMMER', 'FALL']
        seasons_ = ['_'.join(['SEASON', s]) for s in seasons]
        map_ = {k: v for k, v in zip(seasons_, seasons)}

        dataframe = pd.get_dummies(dataframe, columns = ['SEASON'])
        dataframe.rename(map_, axis=1, inplace=True)
    return dataframe

def get_elevation(dataframe, lat='LATITUDE', lon='LONGITUDE', copy=False):
    if copy:
        dataframe = dataframe.copy()

    source = './data/shape_file/elevation_info/elevation.tif'
    if not os.path.exists(source):
        link = 'http://srtm.csi.cgiar.org/srtmdata/'
        raise FileNotFoundError(f'Elevation file not found. \
                                Download file from {link} and place it at {source}.')

    latitude = dataframe[lat].tolist()
    longitude = dataframe[lon].tolist()
    coords = [(lon, lat) for lon, lat in zip(longitude, latitude)]

    with rasterio.open(source) as f:
        vals = list(f.sample(coords))

    vals = np.array(vals).ravel()
    vals[vals == -32768.0] = 0 #sentinel value for missing (usually it's in the sea) values

    dataframe['ELEVATION'] = vals
    print('column ELEVATION has been added to the dataframe.')
    return dataframe


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

def fill_missing_dates(df_original, spatial_column=None, temporal_column=None, method_to_fill_missing='nearest'):
    '''
    df_original (pandas dateframe): dataframe with missing dates
    spatial_column (string): the column name of stations
        since we assume there are multiple dates for the same locatino
    temporal_column (string): the name of column that contains dates
    '''

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
    return df


def remove_incomplete_locations(dataframe, station='SITENUMBER', varname='GAGE_MAX', threshold=0.9):
    '''
    Removes locations based on temporal completeness. Default is 90%, which means we discard
    locations with more than 10% of missing dates.
    We assume there is no missing dates for each location, if you suspect otherwise,
    use fill_missing_dates to preprocess data first.
    '''
    size_ = pd.Series(dataframe.groupby([station])[[varname]].size()) #no NA
    count_ = pd.Series(dataframe.groupby([station])[[varname]].count().values.ravel(), index=size_.index) #include na and everything

    start_counter = len(size_)
    ratio = (count_/size_).to_frame()
    ratio.columns = ['RATIO']

    df_with_ratio =pd.merge(dataframe, ratio, left_on=station, right_index=True)
    df_with_ratio = df_with_ratio[ df_with_ratio['RATIO'] >= threshold]
    df_with_ratio.drop(['RATIO'], axis=1, inplace=True)

    end_counter = len(pd.Series(df_with_ratio.groupby([station])[[varname]].size()))
    print(f'setting threshold to {threshold} has removed {start_counter - end_counter} locations')
    return df_with_ratio

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

def mark_flood_season(original_df, start='2015-10-01', end='2015-12-31',
   time_col='DATE', name='FLOOD_SEASON'):
    '''
    add a dummy variable for flood season.
    This is because the rainfall and flood dynamics might be different on normal days vs. flood seasons.
    use argwhere can be 6 times faster than using a loop.
    '''
    criteria = (original_df[time_col] >= start) & (original_df[time_col] <= end)
    original_df[name] = np.where(criteria, 1, 0)
    return original_df

def add_lag_terms(df, steps=1, groupby_var='SITENUMBER',
    date_var='DATE', variable='GAGE_MAX_DEV', copy=True):
    '''
    simple add np.nan in the first position, and shift everyone back a slot
    This is complicated by the existence of multiple stations.
    Hence, we have to group by sitenumber, and for each set do the aforementioned operations.

    dataframe(pandas dataframe): a spatial temporal dataframe
    steps(int): the number of autoreg term to be generated (default is 1)
    groupby(str): the location/site identifier column. Each station/site should have its own time series default value (groupby)
    variable(str): the name of the target column; default value 'DEV_GAGE_MAX'
    '''
    if date_var not in dataframe:
        raise KeyError(f'cannot find {date_var}. Need to sort values by dates')
        return None

    if groupby_var not in dataframe:
        raise KeyError(f'cannot find {groupby_var}. Need to sort dates by station')
        return None

    if copy:
        dataframe = dataframe.copy()

    dataframe.sort_values([groupby_var, date_var])
    for i in range(steps):
        additional_rows = dataframe.groupby(groupby_var)[variable].shift(i+1).to_frame()
        dataframe = pd.merge(dataframe, additional_rows,
                      left_index=True,
                      right_index=True,
                      suffixes=['', '_MINUS_'+ str(i+1)])
    return dataframe

if __name__ == '__main__':

    # raw_df = pd.read_csv('./data/rainfall_and_flood_10_beta.csv', index_col=0, dtype={'SITENUMBER': str})

    checkout_df = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    checkout_df = mark_flood_season(checkout_df, start='2015-10-01',
        end='2015-12-31')

    ss = transpose_dataframe(checkout_df, start='2010-01-01', end='2010-01-02')

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
    raw_df = get_lag_terms(raw_df, steps=2)
    raw_df.to_csv('./data/check_out.csv') #13.6 -> 6.3
