from datatools import dailyToMonthlyConverter
from SSTcalculator import SSTcalculator
import pandas as pd
from dateutil import relativedelta
from datetime import datetime

class RangeSelector:

    '''
    Generate dataset within given range.
    Incomplete records will be filtered out.
    Optionally, user can combine the precipitation with sst info.
    see the detaile below.

    Args:
        monthly_rainfall_df (pandas dataframe): a pandas dataframe with
        columns at least these: 'STATION', 'YEAR', 'MONTH', 'LONGITUDE', 'LATITUDE', 'PRCP', 'SST'

        in_date (string): the starting year and month eg. '2010-10'
        out_date (string): the ending year and month eg. '2010-10'
    '''

    def __init__(self, monthly_rainfall_df, in_date, out_date):

        self.monthly_rainfall_df = monthly_rainfall_df
        self.in_date = in_date
        self.out_date = out_date
        self.monthly_rainfall_df.DATE = pd.to_datetime(self.monthly_rainfall_df.DATE)
        self._select_complete_records_with_given_range()

    def _select_complete_records_with_given_range(self):

        starting = self.in_date; ending = self.out_date
        print('both begining month and ending month will be included')
        starting = datetime.strptime(str(starting), '%Y-%m')
        ending = datetime.strptime(str(ending), '%Y-%m') + relativedelta.relativedelta(months = 1)

        delta = relativedelta.relativedelta(starting, ending)

        n_months = abs(delta.months)
        n_years = abs(delta.years)
        self.month_count = n_months + n_years*12

        df_w_quanlified_dates = self.monthly_rainfall_df[(self.monthly_rainfall_df.DATE >= starting) \
                                           & (self.monthly_rainfall_df.DATE <= ending)]
        stations_w_complete_records_ = df_w_quanlified_dates.groupby('STATION').count().PRCP == self.month_count
        stations_w_complete_records = list(stations_w_complete_records_.index[stations_w_complete_records_.values])
        self.complete_records_wo_sst = df_w_quanlified_dates[df_w_quanlified_dates.STATION.isin(stations_w_complete_records)]

    def merge_with_sst(self, nc_data_dir, nc_mask_dir):

        cal = SSTcalculator(nc_data_dir = nc_data_dir, nc_mask_dir = nc_mask_dir)
        lats_and_lons = self.complete_records_wo_sst.groupby('STATION').first().reset_index()[['STATION', 'LONGITUDE', 'LATITUDE']]
        list_of_forms = cal.look_up_engine(starting_date = self.in_date,
                                           end_date= self.out_date,
                                           spatial_points = lats_and_lons)

        sst_forms = pd.concat(list_of_forms, axis = 0) #stack up (by row)
        sst_forms = sst_forms.drop(['LONGITUDE','LATITUDE'], axis =1) #drop a column

        temp1 = self.complete_records_wo_sst.set_index(['STATION', 'YEAR', 'MONTH'])
        temp2 = sst_forms.set_index(['STATION', 'YEAR', 'MONTH'])

        merged_table = pd.merge(temp1, temp2, left_index= True, right_index= True).reset_index()

        try:
            merged_table.drop(['Unnamed: 0'], axis =1, inplace = True)
        except KeyError:
            pass

        self.merged_table = merged_table
        return merged_table

    def flip_to_flat_and_wide(self, vars_to_retain = None):

        if vars_to_retain is None:
            vars_to_retain = ['PRCP', 'SST','RANGE_HIGH','RANGE_OVERALL',
                                'RANGE_LOW','TMIN', 'TMAX']

        column_name = []
        for var in vars_to_retain:
            for i in range(self.month_count):
                column_name.append(var + str(i + 1))
        try:
            df_reindexed = self.merged_table.set_index(['STATION', 'DATE'])
        except:
            raise AttributeError('Need to first call merge_with_sst()')

        temp1 = df_reindexed.unstack()[vars_to_retain].values
        temp1_index = df_reindexed.unstack()[vars_to_retain].index
        output = pd.DataFrame(temp1, columns = column_name, index = temp1_index).reset_index()

        return output

if __name__ == '__main__':

    target_dir = '/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/monthly_rainfall.csv'
    nc_mask_dir = '/Users/haigangliu/Dropbox/DataRepository/data_file_new/lsmask.nc'
    nc_data_dir = '/Users/haigangliu/Dropbox/DissertationCode/sst/sst.mnmean.nc'

    save_dir = '/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years.csv'
    save_dir2 = '/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years_flat_and_wide.csv'

    monthly_rain = pd.read_csv(target_dir)

    selector = RangeSelector(monthly_rain, '2011-01', '2015-12')
    merged_table = selector.merge_with_sst(nc_data_dir, nc_mask_dir)
    merged_table_flat_and_wide = selector.flip_to_flat_and_wide()

    merged_table.to_csv(save_dir)
    merged_table_flat_and_wide.to_csv(save_dir2)
