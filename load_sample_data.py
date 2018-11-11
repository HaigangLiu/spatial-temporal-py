import pandas as pd
from utility_functions import get_in_between_dates

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

class SpatialTemporalDataLoader:
    '''
    A base class for rainfall and flood dataloader. User should not call
    this class directly. Use inherited class e.g. LoadFloodDaily instead
    '''
    def __init__(self, dataframe, variables, start, end):

        self.start = start
        self.end = end
        self.list_of_dates = get_in_between_dates(self.start, self.end)
        mask = dataframe.DATE.str.startswith(tuple(self.list_of_dates))
        self.daily_data = dataframe[mask][variables]
        self.sanity_check(self.daily_data)

    def print_message(self, custom_message):
        print('-'*20)
        print('Finished loading...')
        print(f'This is the data from {self.start} to {self.end} in pandas dataframe type')
        print(f'{len(self.list_of_dates)} day(s) or month(s) will be included in the result')
        print(custom_message)
        print(f'This dataframe has {self.daily_data.shape[0]} rows, and {self.daily_data.shape[1]} columns')
        print('This is raw data and thus there might be missing values')
        print('-'*20)

    def sanity_check(self, dataframe):
        if dataframe.empty:
            print('Cannot find data for the give criteria')
            print('Please double check the date information you give')
            print('Some function only works for 2015.')
            raise ValueError(f'data not found for the range between {self.start} and {self.end}')

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
        dataframe = pd.read_csv('./data/with_sst_5_years.csv', index_col=0)
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

class LoadFloodMultipleDays(SpatialTemporalDataLoader):
    def __init__(self, start, end):
        dataframe = pd.read_csv('./data/check_out.csv', index_col=0, dtype={'SITENUMBER':str})
        variables = ['SITENUMBER','LATITUDE','LONGITUDE','DATE','GAGE_MAX', 'ELEVATION', 'PRCP','HISTORICAL_MEDIAN_GAGE_MAX','BASIN']
        super().__init__(dataframe, variables, start, end)

    def load(self):
        custom_message = 'Additional information include max and min temperature and rainfall.\n'
        custom_message2 = 'HISTORICAL_MEDIAN_GAGE_MAX gives a historical reference line'
        print('---- Start loading spatial temporal  data for flood----')
        return super().load(custom_message + custom_message2)

class LoadFloodMultipleMonths(SpatialTemporalDataLoader):
    def __init__(self, start, end):
        dataframe = pd.read_csv('./data/check_out_monthly.csv', index_col=0, dtype={'SITENUMBER':str})
        variables = ['SITENUMBER','LATITUDE','LONGITUDE','DATE','GAGE_MAX', 'ELEVATION', 'PRCP','HISTORICAL_MEDIAN_GAGE_MAX','BASIN']
        super().__init__(dataframe, variables, start, end)

    def load(self):
        custom_message = 'Additional information include max and min temperature and rainfall.\n'
        custom_message2 = 'HISTORICAL_MEDIAN_GAGE_MAX gives a historical reference line'
        print('---- Start loading spatial temporal data for monthly max flood----')
        return super().load(custom_message + custom_message2)

class LoadRainMultipleDays(SpatialTemporalDataLoader):
    def __init__(self, start, end):
        dataframe = pd.read_csv('./data/daily_rainfall_with_region_label.csv', index_col=0)
        variables = ['STATION','LATITUDE','LONGITUDE','PRCP', 'ELEVATION', 'TMAX', 'TMIN','DATE']
        super().__init__(dataframe, variables, start, end)

    def load(self):
        custom_message = 'Additional information include max and min temperature and rainfall.\n'
        print('---- Start loading spatial temporal data for daily rain----')
        return super().load(custom_message)

class LoadRainMultipleMonths(SpatialTemporalDataLoader):

    def __init__(self, start, end):
        dataframe = pd.read_csv('./data/with_sst_5_years.csv', index_col=0, dtype={'SITENUMBER':str})
        variables = ['STATION','LATITUDE','LONGITUDE','DATE','PRCP', 'ELEVATION', 'SST']
        super().__init__(dataframe, variables, start, end)

    def load(self):
        custom_message = 'Additional information include elevation and sea surface temperature (SST).\n'
        print('---- Start loading spatial temporal data for monthly rainfall max ----')
        return super().load(custom_message)

if __name__ == '__main__':

    #spatial
    s1 = LoadRainByDay(2015, 10, 3).load()
    s2 = LoadFloodByDay(2015, 10, 3).load()
    s3 = LoadRainByMonth(2015, 2).load()
    s4 = LoadFloodByMonth(2015, 2).load()
    #spatial temporal
    s5 = LoadFloodMultipleDays('2015-10-01', '2015-10-03').load()
    s6 = LoadFloodMultipleMonths('2015-10', '2015-12').load()
    s7 = LoadRainMultipleDays('2015-10-01', '2015-10-03').load()
    s8 = LoadRainMultipleMonths('2015-10', '2015-12').load()

    print(s1.head())
    print(s2.head())
    print(s3.head())
    print(s4.head())
    print(s5.head())
    print(s6.head())
    print(s7.head())
    print(s8.head())
