import pandas as pd

def load_rainfall_data(option = 'monthly'):

    if option == 'monthly':
        retained_var_list = ['STATION','YEAR', 'MONTH','LATITUDE','LONGITUDE', 'PRCP', 'SST']
        monthly_data = pd.read_csv('./data/with_sst_1_year.csv')[retained_var_list]
        sample_data = monthly_data[(monthly_data.YEAR == 2015) & (monthly_data.MONTH == 10)]
        print('This is the rainfall data on October 2015.')
        print('Additional info includes temperature and SST.')

    elif option == 'daily':
        retained_var_list = ['STATION','ELEVATION', 'LATITUDE', 'LONGITUDE', 'DATE','PRCP', 'TMAX', 'TMIN']
        daily_data =  pd.read_csv('./data/daily_rainfall_with_region_label.csv',
                                  parse_dates=['DATE'],
                                  date_parser= pd.to_datetime)
        sample_data = daily_data.loc[daily_data.DATE == pd.to_datetime('2015-10-03'), retained_var_list]
        print('This is the rainfall data on October 3, 2015.')
        print('Additional info includes temperature (max, min and mean).')
        return sample_data

    else:
        raise ValueError(f'{option} is not available. Only accepts monthly or daily.')
        return None
    print(f'There are {len(sample_data)} rows.')
    return sample_data.reset_index(drop = True)

def load_flood_data(option = 'monthly'):

    if option == 'monthly':
        retained_var_list = ['SITENUMBER', 'YEAR', 'MONTH','LATITUDE', 'LONGITUDE', 'GAGE_MAX']
        data_frame = pd.read_csv('./data/flood_data_1_year.csv', index_col = 0)
        sample_data = data_frame[(data_frame.YEAR == 2015) & (data_frame.MONTH == 10)]
        sample_data = sample_data[retained_var_list].reset_index(drop = True)
        print(f'This is the flood data on October 2015.')
        print(f'Additional info includes location and site number')

    elif option  == 'daily':
        retained_var_list = ['SITENUMBER', 'DATE','LATITUDE', 'LONGITUDE', 'GAGE_MAX']
        daily_data = pd.read_csv('./data/flood_data_daily.csv',
                                dtype = {'SITENUMBER': str, 'GAGE_MAX': float},
                                parse_dates=['DATE'],
                                na_values = ['Eqp'],
                                date_parser= pd.to_datetime)
        daily_data = daily_data.loc[daily_data.DATE == pd.to_datetime('2015-10-03'), retained_var_list]
        sample_data = daily_data[retained_var_list].reset_index(drop = True)
        print(f'This is the flood data on October 3, 2015.')
        print(f'Additional info includes location and site number')

    else:
        raise ValueError(f'options only accepts monthly or daily. {option} is not available.')
        return None
    print(f'There are {len(sample_data)} rows in this dataset.')
    return sample_data

def load_flood_data_spatial_temporal(option = 'five-year'):

    retained_var_list =[ 'SITENUMBER',  'YEAR', 'MONTH', 'LATITUDE', 'LONGITUDE', 'GAGE_MAX']

    if option == 'one-year':
        sample_data = pd.read_csv('./data/flood_data_1_year.csv',
                                  dtype = {'SITENUMBER': str})[retained_var_list]
        more_info = 'year 2015'

    elif option == 'five-year':
        sample_data = pd.read_csv('./data/flood_data_5_years.csv',
                                  dtype = {'SITENUMBER': str})[retained_var_list]
        more_info = '2011 - 2015'

    else:
        raise ValueError(f'only accepts one-month or five-month. {option} is not available.')
        return None

    print(f'This is the rainfall data of {option} : {more_info}. There are {len(sample_data)} records.')
    return sample_data

def load_rainfall_data_spatial_temporal(option = 'five-year'):

    retained_var_list = ['STATION','YEAR', 'MONTH','LATITUDE','LONGITUDE', 'PRCP', 'SST']

    if option == 'one-year':
        sample_data = pd.read_csv('./data/with_sst_1_year.csv')[retained_var_list]
        more_info = 'year 2015'

    elif option == 'five-year':
        sample_data = pd.read_csv('./data/with_sst_5_years.csv')[retained_var_list]
        more_info = '2011 - 2015'

    else:
        raise ValueError(f'only accepts one-month or five-month. {option} is not available.')
        return None
    print(f'This is the rainfall data of {option} : {more_info}. There are {len(sample_data)} records.')
    return sample_data

if __name__ == '__main__':

    s1 = load_flood_data('monthly')
    s2 = load_flood_data('daily')
    s3 = load_rainfall_data('monthly')
    s4 = load_rainfall_data('daily')

    s5 = load_rainfall_data_spatial_temporal('one-year')
    s6 = load_flood_data_spatial_temporal('one-year')
    s7 = load_rainfall_data_spatial_temporal('five-year')
    s8 = load_flood_data_spatial_temporal('five-year')
