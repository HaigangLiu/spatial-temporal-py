import pandas as pd
import numpy as np
import os
from functools import partial

class dailyRainHandler:
    '''
    Compute the monthly maximum rainfall/temperature data based on daily data.
    The range of temperatures will be computed as well.

    Args:
        file_dir (string): the directory where the rainfall records are stored.
        save_dir (string): the directory to save generated monthly form.
    '''
    def __init__(self, file_dir, save_dir):
        self.file_dir = file_dir
        self.save_dir = save_dir
        self.dtype_dict = {
                            'STATION': str,
                            'STATION_NAME': str,
                            'ELEVATION': float,
                            'LATITUDE': float,
                            'LONGITUDE': float,
                            'DATE': str,
                            'PRCP': float,
                            'TMAX': float,
                            'TMIN': float
                           }
        #self.date_parser = partial(pd.to_datetime, format='%Y%m%d')

    def _single_file_handler(self, file_name):

        file_name = self.file_dir + file_name
        assert os.path.exists(file_name), 'Make sure the file exists! Or maybe you missed .csv?'
        data_frame = pd.read_csv(file_name,
                          usecols = list(self.dtype_dict.keys()),
                          dtype = self.dtype_dict,
                         # converters = {'DATE': self.date_parser}, #converter is way slower
                          na_values=['unknown', -9999, 9999] #addtional values treated as nan
                          )

        data_frame['DATE'] = pd.to_datetime(data_frame.DATE, format = "%Y%m%d")
        pivot_table = data_frame.pivot(index = 'DATE',
                                       columns = 'STATION', #use STATION_NAME might cause a duplicate problem
                                       values =['PRCP', 'TMAX', 'TMIN','LONGITUDE',
                                                'LATITUDE','ELEVATION'
                                               ]
                                       )

        max_values = pivot_table.resample('M').max()
        min_values = pivot_table.resample('M').min()

        max_values_ = max_values.stack().reset_index()
        min_values_ = min_values.stack().reset_index()

        output_df = max_values_.copy()

        output_df['RANGE_HIGH'] = max_values_['TMAX'] - min_values_['TMAX']
        output_df['RANGE_LOW'] = max_values_['TMIN'] - min_values_['TMIN']
        output_df['RANGE_OVERALL'] = max_values_['TMAX'] - min_values_['TMIN']
        output_df['RANGE_MID'] = 0.5*(max_values_['TMAX'] + min_values_['TMIN'])

        output_df["YEAR"] = output_df['DATE'].dt.year
        output_df["MONTH"] = output_df['DATE'].dt.month

        return output_df

    def process_all_files(self, new_file_name):
        file_content_list = []
        for file_name in os.listdir(self.file_dir):
            if file_name.endswith('.csv'):
                one_file = self._single_file_handler(file_name)
                print(f'finished processing file {file_name} and converted it into a monthly file.')
                print(f'generated {one_file.shape[0]} records')
                file_content_list.append(one_file)

        output = pd.concat(file_content_list, axis = 0)
        output.to_csv(self.save_dir + new_file_name)
        return output

if __name__ == '__main__':
    raw_daily_rain_file = '/Users/haigangliu/Dropbox/DissertationCode/precipitation/'
    target_dir = '/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/'
    handler = dailyRainHandler(raw_daily_rain_file, target_dir)
    all_files = handler.process_all_files('monthly_rainfall.csv')
