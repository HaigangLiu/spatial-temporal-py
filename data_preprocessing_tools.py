import pandas as pd
import numpy as np

def add_historical_median(dataframe, location_id=None, varname=None, delete_empty_loc=True):
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
        dataframe.set_index(location_id, inplace=True)
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

def simple_imputatation(dataframe, spatial_column=None, temporal_column=None, varname=None):
    '''
    For all locations in the dataset, there is any missing value,
    we will impute the missing value by averaging the time before
    and the time point after.
    '''
    spatial_column = 'SITENUMBER' if spatial_column is None else spatial_column
    temporal_column = 'DATE' if temporal_column is None else temporal_column
    varname = 'GAGE_MAX' if varname is None else varname

    dataframe.sort_values(['SITENUMBER', 'DATE'], inplace=True)
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

if __name__ == '__main__':
    from SampleDataLoader import load_rainfall_data
    # rainfall = load_rainfall_data('monthly')

    file_ = pd.read_csv('./data/rainfall_and_flood_10_beta.csv', index_col=0, dtype={'SITENUMBER':str})
    print(file_.head())
    median_added = add_historical_median(file_, varname='GAGE_MAX', location_id= 'SITENUMBER', delete_empty_loc=False)
    simple_imputatation(median_added, spatial_column='SITENUMBER', varname='GAGE_MAX')
