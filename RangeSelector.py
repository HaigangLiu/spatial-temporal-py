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
        monthly_df (pandas dataframe): a pandas dataframe with
        columns at least these: self.unique_column_name, 'YEAR', 'MONTH', 'LONGITUDE', 'LATITUDE', 'PRCP', 'SST'

        in_date (string): the starting year and month eg. '2010-10'
        out_date (string): the ending year and month eg. '2010-10'
    '''

    def __init__(self, monthly_df, in_date, out_date, unique_column_name = 'STATION', response_var = 'PRCP'):

        self.unique_column_name = unique_column_name
        self.monthly_df = monthly_df

        self.response_var = response_var
        self.in_date = in_date
        self.out_date = out_date

        self.monthly_df.DATE = pd.to_datetime(self.monthly_df.DATE)
        self._select_df_within_given_range()

    def _select_df_within_given_range(self):

        starting = self.in_date; ending = self.out_date
        print('both begining month and ending month will be included')
        starting = datetime.strptime(str(starting), '%Y-%m')
        ending = datetime.strptime(str(ending), '%Y-%m') + relativedelta.relativedelta(months = 1)

        delta = relativedelta.relativedelta(starting, ending)

        n_months = abs(delta.months)
        n_years = abs(delta.years)
        self.month_count = n_months + n_years*12

        df_w_quanlified_dates = self.monthly_df[(self.monthly_df.DATE >= starting) \
                                           & (self.monthly_df.DATE <= ending)]
        stations_w_complete_records_ = df_w_quanlified_dates.groupby(self.unique_column_name).count()[self.response_var] == self.month_count
        stations_w_complete_records = list(stations_w_complete_records_.index[stations_w_complete_records_.values])
        self.complete_records_wo_sst = df_w_quanlified_dates[df_w_quanlified_dates[self.unique_column_name].isin(stations_w_complete_records)]

    def add_sst_info(self, nc_data_dir, nc_mask_dir):

        cal = SSTcalculator(nc_data_dir = nc_data_dir, nc_mask_dir = nc_mask_dir)
        lats_and_lons = self.complete_records_wo_sst.groupby(self.unique_column_name).first().reset_index()[[self.unique_column_name, 'LONGITUDE', 'LATITUDE']]
        list_of_forms = cal.look_up_engine(starting_date = self.in_date,
                                           end_date= self.out_date,
                                           spatial_points = lats_and_lons)

        sst_forms = pd.concat(list_of_forms, axis = 0) #stack up (by row)
        sst_forms = sst_forms.drop(['LONGITUDE','LATITUDE'], axis =1) #drop a column

        temp1 = self.complete_records_wo_sst.set_index([self.unique_column_name, 'YEAR', 'MONTH'])
        temp2 = sst_forms.set_index([self.unique_column_name, 'YEAR', 'MONTH'])

        merged_table = pd.merge(temp1, temp2, left_index= True, right_index= True).reset_index()

        self.merged_table = merged_table
        return merged_table

    def flip_to_flat_and_wide(self, vars_to_retain):

        if vars_to_retain is None:
            vars_to_retain = ['PRCP', 'SST','RANGE_HIGH','RANGE_OVERALL', 'RANGE_LOW','TMIN', 'TMAX']

        column_name = []

        for var in vars_to_retain:
            for i in range(self.month_count):
                column_name.append(var + str(i + 1))

        try:
            location_info = self.merged_table[[self.unique_column_name, 'LATITUDE', 'LONGITUDE']]
            df_reindexed = self.merged_table.set_index([self.unique_column_name, 'DATE'])

        except AttributeError:
            print('skipped the step of calculating sst')
            location_info = self.complete_records_wo_sst[[self.unique_column_name, 'LATITUDE', 'LONGITUDE']]
            df_reindexed = self.complete_records_wo_sst.set_index([self.unique_column_name, 'DATE'])

        unique_locations = location_info.groupby(self.unique_column_name).first().reset_index().copy()

        temp1 = df_reindexed.unstack()[vars_to_retain].values
        temp1_index = df_reindexed.unstack()[vars_to_retain].index

        output = pd.DataFrame(temp1, columns = column_name, index = temp1_index).reset_index()
        output.set_index(self.unique_column_name, inplace = True)
        unique_locations.set_index(self.unique_column_name, inplace = True)

        output_with_lat_lon = output.merge(unique_locations, left_index = True, right_index = True).reset_index()
        return output_with_lat_lon

if __name__ == '__main__':

    # ---- working with rainfall data ------

    nc_mask_dir = '/Users/haigangliu/Dropbox/DataRepository/data_file_new/lsmask.nc'
    nc_data_dir = '/Users/haigangliu/Dropbox/DissertationCode/sst/sst.mnmean.nc'

    target_dir_rain = './data/monthly_rainfall.csv'
    monthly_rain = pd.read_csv(target_dir_rain)

    selector =  RangeSelector(monthly_rain,
                               in_date = '2015-01',
                               out_date = '2015-12',
                               unique_column_name = 'STATION',
                               response_var = 'PRCP')\

    one_year_rain = selector.add_sst_info(nc_data_dir, nc_mask_dir)
    one_year_rain_fw = selector.flip_to_flat_and_wide(vars_to_retain = ['SST','PRCP','TMAX','TMIN'])


    selector_2 =  RangeSelector(monthly_rain,
                               in_date = '2011-01',
                               out_date = '2015-12',
                               unique_column_name = 'STATION',
                               response_var = 'PRCP')\

    five_year_rain = selector_2.add_sst_info(nc_data_dir, nc_mask_dir)
    five_year_rain_fw = selector_2.flip_to_flat_and_wide(vars_to_retain = ['SST','PRCP','TMAX','TMIN'])


    # ---- working with flood data ------

    target_dir_flood = './data/flood_data_5_years.csv'
    monthly_flood = pd.read_csv(target_dir_flood)

    five_year =  RangeSelector(monthly_flood,
                               in_date = '2011-01',
                               out_date = '2015-12',
                               unique_column_name = 'SITENUMBER',
                               response_var = 'GAGE_MAX')\
                .flip_to_flat_and_wide(vars_to_retain = ['GAGE_MAX'])
    five_year.to_csv('./data/flood_5_years_flat_and_wide.csv')

    one_year =  RangeSelector(monthly_flood,
                               in_date = '2015-01',
                               out_date = '2015-12',
                               unique_column_name = 'SITENUMBER',
                               response_var = 'GAGE_MAX')\
                .flip_to_flat_and_wide(vars_to_retain = ['GAGE_MAX'])
    one_year.to_csv('./data/flood_1_year_flat_and_wide.csv')



