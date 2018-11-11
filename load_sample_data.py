import pandas as pd
class SpatialDataLoader:
    '''
    A base class for rainfall and flood dataloader. User should not call
    this class directly. Use inherited class e.g. LoadFloodDaily instead
    '''
    def __init__(self, dataframe, variables, year, month, day=None):

        self.date = self._make_date(year, month, day)
        self.dataframe = dataframe
        self.daily_data = self.dataframe[self.dataframe.DATE.str.startswith(self.date)][variables]
        self.sanity_check(self.daily_data)

    def _make_date(self, year, month, day):
        month = str(month).zfill(2)
        year = str(year)

        if day is None:
            date = '-'.join([year, month])
        else:
            day = str(day).zfill(2)
            date = '-'.join([year, month, day])
        return date

    def print_message(self, custom_message):
        print('-'*20)
        print('Finished loading...')
        print(f'This is the data on {self.date} in pandas dataframe type')
        print(custom_message)
        print(f'This dataframe has {self.daily_data.shape[0]} rows, and {self.daily_data.shape[1]} columns')
        print('This is raw data and thus there might be missing values')
        print('-'*20)

    def sanity_check(self, dataframe):
        if dataframe.empty:
            print('Cannot find data for the give criteria')
            print('Please double check the date information you give')
            print('Some function only works for 2015.')
            raise ValueError(f'data not found for {self.date}')

    def load(self, custom_message):
        self.print_message(custom_message)
        return self.daily_data.reset_index(drop=True)

class LoadFloodByDay(SpatialDataLoader):

    def __init__(self, year, month, day):
        dataframe = pd.read_csv('./data/check_out.csv', index_col=0, dtype={'SITENUMBER':str})
        variables = ['SITENUMBER','LATITUDE','LONGITUDE','GAGE_MAX', 'ELEVATION', 'PRCP','HISTORICAL_MEDIAN_GAGE_MAX']
        super().__init__(dataframe, variables, year, month, day)

    def load(self):
        custom_message = 'Additional information include max and min temperature and rainfall.\n'
        custom_message2 = 'HISTORICAL_MEDIAN_GAGE_MAX gives a historical reference line'
        print('---- Start loading daily flood data ----')
        return super().load(custom_message + custom_message2)

class LoadRainByDay(SpatialDataLoader):

    def __init__(self, year, month, day):
        dataframe = pd.read_csv('./data/daily_rainfall_with_region_label.csv', index_col=0)
        variables = ['STATION','LATITUDE','LONGITUDE','PRCP', 'ELEVATION', 'TMAX', 'TMIN']
        super().__init__(dataframe, variables, year, month, day)

    def load(self):
        custom_message = 'Additional information include max and min temperature and elevation'
        print('----  Start loading daily rainfall data ----')
        return super().load(custom_message)

class LoadRainByMonth(SpatialDataLoader):

    def __init__(self, year, month, day=None):
        dataframe = pd.read_csv('./data/with_sst_1_year.csv', index_col=0)
        variables = ['STATION','LATITUDE','LONGITUDE','PRCP', 'ELEVATION', 'SST']
        super().__init__(dataframe, variables, year, month, day)

    def load(self):
        custom_message = 'Additional information include elevation and sea surface temperature'
        print('----  Start loading monthly (maximum) rainfall data ----')
        return super().load(custom_message)

class LoadFloodByMonth(SpatialDataLoader):

    def __init__(self, year, month, day=None):
        dir_ = './data/check_out_monthly.csv'
        dataframe = pd.read_csv(dir_, index_col=0, dtype={'SITENUMBER': str})
        variables = ['SITENUMBER','LATITUDE','LONGITUDE','GAGE_MAX', 'ELEVATION', 'PRCP','HISTORICAL_MEDIAN_GAGE_MAX']
        super().__init__(dataframe, variables, year, month, day=None)

    def load(self):
        custom_message = 'Additional information include max and min temperature and rainfall.\n'
        custom_message2 = 'HISTORICAL_MEDIAN_GAGE_MAX gives a historical reference line'
        print('---- Start loading monthly (maximum) flood data ----')
        return super().load(custom_message + custom_message2)

if __name__ == '__main__':

    s1 = LoadRainByDay(2015, 10, 3).load()
    print(s1.head())
    s2 = LoadFloodByDay(2015, 10, 3).load()
    print(s2.head())
    s3 = LoadRainByMonth(2015, 2).load()
    print(s3.head())
    s4 = LoadFloodByMonth(2015, 2).load()
    print(s4.head())
