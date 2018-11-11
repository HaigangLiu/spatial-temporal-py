import pandas as pd

BASE_COLUMN_PRCP = ['STATION','LATITUDE','LONGITUDE','PRCP']
BASE_COLUMN_FLOOD = ['SITENUMBER','LATITUDE','LONGITUDE','GAGE_MAX']

def generate_information(dataframe, additional_variables, year, month, day=None):
    '''
    a bunch of print functions of give information of the dataframe
    '''
    print('-'*20)
    if day is None:
        print(f'This is the data on {year}-{month}. (monthly)')
    else:
        print(f'This is the data on {year}-{month}-{day}. (daily)')

    print(f"the additional variables include {', '.join(additional_variables)}")
    print(f'This dataframe has {dataframe.shape[0]} rows, and {dataframe.shape[1]} columns')
    print('This is raw data and thus there might be missing values')
    print('-'*20)

def load_monthly_rainfall(additional_variables=None, year=2015, month=10):
    '''
    load a data frame of rainfall for a given month, Oct 2015 by default.
    Some other information include latitude and longitude. Also sea surface temperature is provided.
    '''
    target_dir = './data/with_sst_1_year.csv'
    additional_variables = ['ELEVATION', 'SST'] if additional_variables is None else additional_variables

    columns_selected = BASE_COLUMN_PRCP.copy()
    columns_selected.extend(additional_variables)

    data_all = pd.read_csv(target_dir, index_col=0)
    monthly_data = data_all[(data_all.YEAR==year)&(data_all.MONTH == month)][columns_selected
    ]

    if not monthly_data.empty:
        generate_information(monthly_data, additional_variables, year, month, day=None)
        return monthly_data.reset_index(drop=True)
    else:
        raise ValueError(f'No data found for {year}-{month}')

def load_daily_rainfall(additional_variables=None, year=2015, month=10, day=3):
    '''
    load a data frame of rainfall for a given day, Oct 3 2015 by default.
    Some other information include latitude and longitude. Also daily max and min temperature are provided.
    However, TMAX and TMIN are more often than not missing.
    '''
    target_dir = './data/daily_rainfall_with_region_label.csv'
    data_all = pd.read_csv(target_dir, index_col=0).reset_index(drop=True)

    month = str(month).zfill(2)
    day = str(day).zfill(2)
    year = str(year)
    date = '-'.join([year, month, day])

    additional_variables = ['ELEVATION', 'TMAX', 'TMIN'] if additional_variables is None else additional_variables
    columns_selected = BASE_COLUMN_PRCP.copy()
    columns_selected.extend(additional_variables)
    daily_data = data_all[data_all.DATE.str.startswith(date)][columns_selected]

    if not daily_data.empty:
        generate_information(daily_data, additional_variables, year, month, day)
        return daily_data.reset_index(drop=True)
    else:
        raise ValueError(f'No data found for {year}-{month}-{day}')

def load_monthly_flood(year=2015, month=10):
    # try use the data from the checkout file

    print('Rainfall information are given as well, as a covariate')
    target_dir = './data/flood_data_1_year.csv'
    columns_selected = BASE_COLUMN_FLOOD.copy()

    data_all = pd.read_csv(target_dir, index_col=0, dtype={'SITENUMBER': str})
    sample_data = data_all[(data_all.YEAR == year) & (data_all.MONTH == month)]
    sample_data = sample_data[BASE_COLUMN_FLOOD].reset_index(drop = True)
    return sample_data

def load_daily_flood(year=2015, month=10, day=3, additional_variables=None):

    month = str(month).zfill(2)
    day = str(day).zfill(2)
    year = str(year)
    date = '-'.join([year, month, day])

    target_dir = './data/check_out.csv'
    data_all = pd.read_csv(target_dir, index_col=0, dtype={'SITENUMBER': str})

    columns_selected = BASE_COLUMN_FLOOD.copy()
    additional_variables = ['ELEVATION', 'BASIN', 'PRCP'] if additional_variables is None else additional_variables
    columns_selected.extend(additional_variables)

    daily_data = data_all[data_all.DATE.str.startswith(date)][columns_selected]
    if not daily_data.empty:
        generate_information(daily_data, additional_variables, year, month, day)
        return daily_data.reset_index(drop=True)
    else:
        print('-'*20)
        print(f'This is the rainfall data on {year}-{month}-{day} (daily maximum)')
        print(f"the additional variables include {', '.join(additional_variables)}")
        print('This is raw data and thus there might be missing values')
        print('-'*20)
        raise  ValueError(f'No data found for {year}-{month}-{day}')

def load_rainfall_multiple_days():
    pass

def load_flood_multiple_days():
    pass

# print(pd.read_csv('./data/daily_rainfall_with_region_label.csv')['DATE'])
# print(load_daily_rainfall().head())
print(load_monthly_rainfall().head())

print(load_daily_rainfall().head())
print(load_daily_flood().head())
