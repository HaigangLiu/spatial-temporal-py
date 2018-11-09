import pandas as pd


COLUMN_PRCP = ['STATION','LATITUDE','LONGITUDE','PRCP']
COLUMN_FLOOD = ['SITENUMBER','LATITUDE','LONGITUDE','GAGE_MAX']

def load_monthly_rainfall(additional_variables=None, year=2015, month=10):

    target_dir = './data/with_sst_1_year.csv'
    additional_variables = ['ELEVATION', 'SST'] if additional_variables is None else additional_variables

    columns_selected = COLUMN_PRCP.copy()
    columns_selected.extend(additional_variables)

    data_all = pd.read_csv(target_dir, index_col=0)
    monthly_data = data_all[(data_all.YEAR==year)&(data_all.MONTH == month)][columns_selected
    ]

    if not monthly_data.empty:
        print('-'*20)
        print(f'This is the rainfall data on {year}-{month}. (monthly cumsum)')
        print(f"the additional variables include {', '.join(additional_variables)}")
        print('This is raw data and thus there might be missing values')
        print('-'*20)
        return monthly_data.reset_index(drop=True)
    else:
        raise ValueError(f'No data found for {year}-{month}')

def load_daily_rainfall(additional_variables=None, year=2015, month=10, day=3):

    target_dir = './data/daily_rainfall_with_region_label.csv'
    data_all = pd.read_csv(target_dir, index_col=0).reset_index(drop=True)

    month = str(month).zfill(2)
    day = str(day).zfill(2)
    year = str(year)
    date = '-'.join([year, month, day])

    additional_variables = ['ELEVATION', 'TMAX', 'TMIN'] if additional_variables is None else additional_variables
    columns_selected = COLUMN_PRCP.copy()
    columns_selected.extend(additional_variables)
    daily_data = data_all[data_all.DATE.str.startswith(date)][columns_selected]

    if not daily_data.empty:
        print('-'*20)
        print(f'This is the rainfall data on {year}-{month}-{day} (daily maximum)')
        print(f"the additional variables include {', '.join(additional_variables)}")
        print('This is raw data and thus there might be missing values')
        print('-'*20)
        return daily_data.reset_index(drop=True)
    else:
        raise ValueError(f'No data found for {year}-{month}-{day}')


def load_monthly_flood(year=2015, month=10):
    # try use the data from the checkout file
    target_dir = './data/flood_data_1_year.csv'
    columns_selected = COLUMN_FLOOD.copy()

    data_all = pd.read_csv(target_dir, index_col=0, dtype={'SITENUMBER': str})
    sample_data = data_all[(data_all.YEAR == year) & (data_all.MONTH == month)]
    sample_data = sample_data[COLUMN_FLOOD].reset_index(drop = True)
    return sample_data

def load_daily_flood():
    pass

def load_rainfall_multiple_days():
    pass

def load_flood_multiple_days():
    pass

# print(pd.read_csv('./data/daily_rainfall_with_region_label.csv')['DATE'])
# print(load_daily_rainfall().head())
print(load_monthly_flood().head())

