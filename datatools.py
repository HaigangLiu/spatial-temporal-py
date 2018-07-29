import pandas as pd
import numpy as np
import os

class dailyToMonthlyConverter:
    '''
    Convert daily data into monthly data based on given aggregation operations.

    There are three kinds of variables:

        1. STATION and DATE: used as index. Users need to specify
         what they are called in the dataframe, but do not need to
         speficy the aggregation method.
        2 Variable like PRCP: the variable change over time. We
        need to aggregate over them. And the aggregation operation
        must be specified. Preferably as numpy functions like np.max.
        3. Variables like ELEVATION: the variable that does not
         change over time. You still need to specify an operation,
         but anything would be fine.

    Args:
        file_dir (string): the directory where the rainfall records are stored.
        dtype_dict (dict): a dictionary of data type
        aggregation_operations(dict): a dictionary of aggregation operations from daily record to monthly.
            Example: {'NEW_PRCP': [np.max, 'PRCP']}
            NEW_PRCP is the new variable name, and np.max is the
            name of aggregation, and 'PRCP' is the original name in the daily record
        station_column (str): name of the columnn of station name.
        date_column (str): name of the columnn of station name.
    '''
    def __init__(self, file_dir,  dtype_dict = None, aggregation_operations = None, station_column = None, date_column = None):

        self.file_dir = file_dir
        self.file_list = [file_name for file_name in os.listdir(self.file_dir) if file_name.endswith('.csv')]

        if dtype_dict is None:
            self.dtype_dict = {'STATION': str,
                                'ELEVATION': float,
                                'LATITUDE': float,
                                'LONGITUDE': float,
                                'DATE': str,
                                'PRCP': float,
                                'TMAX': float,
                                'TMIN': float
                               }
        else:
            self.dtype_dict = dtype_dict

        if aggregation_operations is None:
            self.aggregation_operations = {'PRCP': [np.max, 'PRCP'],
                            'TMAX': [np.max,'TMAX'],
                            'TMIN': [np.min, 'TMIN'],
                            'TMAX_MIN': [np.min, 'TMAX'],
                            'TMIN_MAX': [np.max, 'TMIN'],
                            'ELEVATION': [np.max, 'ELEVATION']
                            }
        else:
            self.aggregation_operations = aggregation_operations

        if station_column is None:
            self.station_column = 'STATION'
            assert 'STATION' in list(self.dtype_dict.keys()), 'this column should be part of dtype_dict'
        else:
            self.station_column = station_column

        if date_column is None:
            self.date_column = 'DATE'
            assert 'DATE' in list(self.dtype_dict.keys()), 'this column should be part of dtype_dict'
        else:
            self.date_column = date_column

    def single_file_handler(self, file_name):

        '''
        Args: file_name (str): the name of a file. Must specify the extension.
        '''

        file_name = self.file_dir + file_name
        assert os.path.exists(file_name), 'Make sure the file exists! Or maybe you missed .csv?'
        data_frame = pd.read_csv(file_name,
                          usecols = list(self.dtype_dict.keys()),
                          dtype = self.dtype_dict,
                          na_values=['unknown', -9999, 9999, 'Eqp']
                          )

        data_frame[self.date_column] = pd.to_datetime(data_frame.DATE)

        value_columns = [col for col in list(self.dtype_dict.keys()) if col  not in [self.date_column, self.station_column]]

        pivot_table = data_frame.pivot(index = self.date_column,
                                       columns = self.station_column, #use STATION_NAME might cause a duplicate problem
                                       values = value_columns
                                       )

        other_cols = [col for col in value_columns if col not in list(self.aggregation_operations.keys())]
        output = pivot_table.resample('M').max().stack()[other_cols]
        for new_col, ops_and_colname in self.aggregation_operations.items():
            ops, col = ops_and_colname
            output[new_col] = pivot_table.resample('M').apply(ops).stack()[col]

        output_df  = pd.DataFrame(output.reset_index())
        output_df["YEAR"] = output_df['DATE'].dt.year
        output_df["MONTH"] = output_df['DATE'].dt.month
        return output_df

    def multiple_file_handler(self, file_list_ = None, multiprocessing = True):
        '''
        Args:
            multiprocessing (boolean): Use multiple processing or not. The default is on.
            file_list_: a list of files names. If not specified, all csv file in this directory will be parsed

        '''
        if file_list_ is None:
            file_list_ = self.file_list

        if multiprocessing:
            import concurrent.futures
            executor = concurrent.futures.ProcessPoolExecutor(max_workers = 8)
            result_generator = executor.map(self.single_file_handler, file_list_)
            output = pd.concat(list(result_generator))

        else:
            for file_name in file_list_:
                one_file = self.single_file_handler(file_name)
                print(f'finished processing file {file_name} and generated {one_file.shape[0]} records')
                file_content_list.append(one_file)
            output = pd.concat(file_content_list, axis = 0)
        return output

if __name__ == '__main__':

    # ----- generate data for precipitation ------
    dir_to_raw_daily_files = '/Users/haigangliu/Dropbox/DissertationCode/precipitation/'
    handler = dailyToMonthlyConverter(dir_to_raw_daily_files)
    single = handler.single_file_handler('829078.csv') #single file
    monthly_rain_output = handler.multiple_file_handler(multiprocessing = True)

    monthly_rain_output['RANGE_HIGH'] = monthly_rain_output['TMAX'] - monthly_rain_output['TMAX_MIN']
    monthly_rain_output['RANGE_LOW'] = monthly_rain_output['TMIN_MAX'] - monthly_rain_output['TMIN']
    monthly_rain_output['RANGE_OVERALL'] = monthly_rain_output['TMAX'] - monthly_rain_output['TMIN']
    monthly_rain_output['RANGE_MID'] = 0.5*monthly_rain_output['TMAX'] + monthly_rain_output['TMIN']
    monthly_rain_output.drop(['TMAX_MIN', 'TMIN_MAX'], inplace = True, axis = 1)

    # note that user can pass in np.ptp function to calculate
    #range, but doing that is mysteriously slow (24 sec for range
       # vs 1 sec for max or min). That's why this operation is
       #  done manually like this.

    # ----- generate data for flood ------
    converter = dailyToMonthlyConverter('./data/',
        dtype_dict = {'LATITUDE': float,
                      'LONGITUDE': float,
                      'DATE': str,
                      'GAGE_MAX': float,
                      'SITENUMBER': str},
        aggregation_operations = {'GAGE': [np.max, 'GAGE_MAX']},
        station_column = 'SITENUMBER',
        date_column = 'DATE')

    df_out = converter.single_file_handler('flood_data_daily.csv')

    five_years_flood = df_out[df_out.YEAR.isin([ 2011, 2012, 2013, 2014, 2015])]
    one_year_flood = df_out[df_out.YEAR == 2015]

    five_years_flood.to_csv('./data/flood_data_5_years.csv')
    one_year_flood.to_csv('./data/flood_data_5_years.csv')
