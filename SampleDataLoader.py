import pandas as pd

pd.read_csv('./data/NWISMapperExport.csv')


def load_sample_data_spatial(data_type = 'rain'):

    if data_type = 'rain':

        print('This is the rainfall data on October 2015 with location information along with temperature and sea surface temperature data')
        monthly_rain = pd.read_csv('./data/with_sst_1_year.csv', index_col = 0)[['STATION','MONTH','YEAR','LATITUDE','LONGITUDE', 'PRCP', 'SST',]]
        sample_data = monthly_rain[(monthly_rain.YEAR == 2015) &(monthly_rain.MONTH == 10)].reset_index(drop = True)
    elif:
        pass

    else:
        raise ValueError('data_type only allow either rain or flood')
    return sample_data

def load_sample_data_spatial_temporal(flat_and_wide = False):
    print('This is the rainfall data from 2010 Januaray to 2015 December with location information')
    if flat_and_wide:
        sample_data = pd.read_csv('./data/with_sst_5_years_flat_and_wide.csv', index_col = 0)
    else:
        sample_data = pd.read_csv('./data/with_sst_5_years.csv', index_col = 0)
    return sample_data
