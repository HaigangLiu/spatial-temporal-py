import pandas as pd
import numpy as np
from utility_functions import get_in_between_dates

def fill_missing_dates(df_original, location_column, date_column, vars_to_propagate=None):
    '''
    df_original (pandas dateframe): dataframe with missing dates
    location_column (string): the column name of stations
        since we assume there are multiple dates for the same locatino
    date_column (string): the name of column that contains dates
    vars_to_propagate (list): specify the variables that does not change with time. e.g. (latitude, longitude, elevations etc.)
        Not specifying this may cause severe missing data issues
    '''
    df_original.reset_index(inplace=True, drop=True)

    unique_days = get_in_between_dates(df_original[date_column].min(), df_original[date_column].max())
    unique_locs = pd.unique(df_original[location_column])

    number_of_locs = len(unique_locs)
    number_of_days = len(unique_days)

    df_new = pd.DataFrame(columns=['SITENUMBER_x', 'DATE_x'])
    df_new['SITENUMBER_x'] = np.repeat(unique_locs, number_of_days)
    df_new['DATE_x'] = np.tile(unique_days, number_of_locs)

    if vars_to_propagate is not None:
        summary_file = df_original.groupby(location_column).first().reset_index() #copy
        df_new.set_index('SITENUMBER_x', inplace=True)
        for idx, row in summary_file.iterrows():
            values = row[vars_to_propagate]
            identifier = row[location_column]
            for var_to_propagate in vars_to_propagate:
                df_new.loc[identifier, var_to_propagate] = values[var_to_propagate]
        df_new.reset_index(inplace=True)

    output = pd.merge(df_new, df_original, how='left', left_on=['SITENUMBER_x', 'DATE_x'], right_on=[location_column, date_column])

    drop_list = [column for column in output.columns if column.endswith('_y')]#clean up
    drop_list.extend([date_column, location_column])
    output.drop(drop_list, inplace=True, axis=1)
    output.columns = [column.replace('_x', '') for column in output.columns]

    return output

if __name__ == '__main__':

    test_file = pd.read_csv('./data/combined_rain_flood.csv', index_col=0, dtype={'SITENUMBER': str})
    combined = fill_missing_dates(test_file, 'SITENUMBER', 'DATE', ['LATITUDE','LONGITUDE','MEDIAN_HISTORICAL','ELEVATION'])
    print(combined.head())
