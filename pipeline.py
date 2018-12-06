import pandas as pd
import pickle
import numpy as np
from merge_rain_and_flood import Merger
from data_preprocessing_tools import fill_missing_dates, filter_missing_locations, get_historical_median, get_elevation, get_watershed, get_basin_from_watershed, get_autoregressive_terms

from_csv = False #just do it once
if from_csv:
    dir_flood = './demo/sc_flood_20100101-20161231.txt'
    df_flood = pd.read_csv(dir_flood, sep='\t',  dtype={'SITENUMBER':str})
    df_flood = df_flood[~(df_flood.SITENUMBER == '021989784')]
    df_flood = df_flood[~(df_flood.SITENUMBER == '021989791')]
    df_flood = df_flood[~(df_flood.SITENUMBER == '0219897993')]

    df_flood['GAGE_MAX'] = pd.to_numeric(df_flood['GAGE_MAX'])
    df_flood['GAGE_MIN'] = pd.to_numeric(df_flood['GAGE_MAX'])
if from_csv:
    dir_rain = './demo/SC_20050101-20170627-19b7.txt'
    df_rain = pd.read_csv(dir_rain, sep=' ')

if from_csv:
    store = {'rain': df_rain, 'flood': df_flood}
    import pickle
    with open('./demo/rain_flood.pickle', 'wb') as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(df_rain)

with open('./demo/rain_flood.pickle', 'rb') as f:
    dict_result = pickle.load(f)
df_rain = dict_result['rain']
df_flood = dict_result['flood']


operation_list = [fill_missing_dates, filter_missing_locations,
get_historical_median, get_elevation, get_watershed, get_basin_from_watershed, get_autoregressive_terms]

for op in operation_list:
    df_flood = op(df_flood)

df_combined = Merger(df_flood, df_rain, '2010-01-01', '2016-12-31').run()
store = {'rain': df_rain, 'flood': df_flood, 'both': df_combined}
with open('./demo/rain_flood.pickle', 'wb') as f:
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)


