import numpy as np
import pandas as pd
import pickle
from pandas.tseries.frequencies import to_offset

def load_data(option='rain', freq=None, agg_ops=None):
    """
    load rainfall or flood data used in the research. Users can choose different frequencies as long as it is wider than original.
    For instance, for rainfall data it's ok to do anually aggregation.
    Users can also customize the aggregation operation by passing a dict.

    Args:
        option(string): only allow rain (include rain and sst, monthly) or flood (include rain and flood, daily).
        freq(string): frequency.(only allow downsampling, upsampling will throw error)
        agg_ops(dict): will be used to update the default aggregation operations.
    """
    DEFAULT_OPS = {'flood':
              {'BASIN': lambda series: series[series.notnull()][0],
              'STATIONNAME': lambda series: series[series.notnull()][0],
              'LATITUDE': np.nanmax,
              'LONGITUDE': np.nanmax,
              'PRCP':np.sum,
              'GAGE_MAX_DEV': np.sum},
              'rain':
              {'LATITUDE': lambda series: series[series.notnull()][0],
               'LONGITUDE': lambda series: series[series.notnull()][0],
               'PRCP': np.nansum,
               'SST':np.nansum,
               'ELEVATION': np.nansum,
               'TMAX': np.max,
               'TMIN': np.min}
               }

    if option == 'flood':
        default_freq = '1D'
        with open('./demo/sample.pickle', 'rb') as f:
            raw_data = pickle.load(f)
            raw_data['DATE'] = pd.to_datetime(raw_data['DATE'])

    elif option == 'rain':
        default_freq = '1M'
        with open('./demo/sample_sst_and_rain.pickle', 'rb') as f:
            raw_data = pickle.load(f)
            raw_data['DATE'] = pd.to_datetime(raw_data['DATE'])
    else:
        raise ValueError(f'{option} not available. Option could only be either rain or flood')

    if freq is None:
        if agg_ops is not None:
            print('agg_ops will be ignored since frequency is not specified')
        return raw_data

    else:
        common_dt = pd.to_datetime("2000-01-01")
        default_time_unit = common_dt + to_offset(default_freq)
        user_input_time_unit = common_dt + to_offset('1' + freq)

        if default_time_unit > user_input_time_unit:
            raise ValueError('cannot do aggregation since user defined freq is smaller than original')
        elif default_time_unit == user_input_time_unit:
            return raw_data
        else:
            operation_dict = DEFAULT_OPS.get(option)
            if agg_ops is not None:
                operation_dict.update(agg_ops)

            def station_wise_agg(stational_df, freq, agg_ops):
                idx_removed_df = stational_df.reset_index(drop=True, level=0)
                return idx_removed_df.resample(freq).agg(agg_ops)

            raw_data.set_index(['SITENUMBER','DATE'], inplace=True)
            agg_data = raw_data.groupby(level=0).apply(station_wise_agg, freq=freq, agg_ops=operation_dict)
            agg_data.reset_index(inplace=True)
            return agg_data

if __name__ == '__main__':

    case_1 = load_data(option='rain') #monthly rain
    case_2 = load_data(option='flood') #daily flood

    case_3 = load_data(option='rain', freq='A') #agg op
    case_4 = load_data(option='flood', freq='A') # agg op

    case_5 = load_data(option='rain', freq='A', agg_ops={'SST':np.max}) #agg op
    case_6 = load_data(option='flood', freq='A', agg_ops={'PRCP':np.max}) #agg op

    case_7 = load_data(option='rain', freq='T') # throw error
    case_8 = load_data(option='flood', freq='T') # throw error
