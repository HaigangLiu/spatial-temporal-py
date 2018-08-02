from SSTcalculator import SSTcalculator
import pandas as pd
from dateutil import relativedelta
from datetime import datetime
import os

class PreprocessingPipeline:

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

    def __init__(self, data_frame, unique_column_name = 'STATION', response_var = 'PRCP'):

        self.data_frame = data_frame
        self.unique_column_name = unique_column_name
        self.response_var = response_var
        self.data_frame.DATE = pd.to_datetime(self.data_frame.DATE)

    def select_given_range(self, in_date, out_date):

        print('both begining month and ending month will be included')

        in_date = datetime.strptime(str(in_date), '%Y-%m')
        out_date = datetime.strptime(str(out_date), '%Y-%m') + relativedelta.relativedelta(months = 1)

        delta = relativedelta.relativedelta(in_date, out_date)

        n_months = abs(delta.months)
        n_years = abs(delta.years)
        self.month_count = n_months + n_years*12

        df_w_quanlified_dates = self.data_frame[(self.data_frame.DATE >= in_date) \
                                           & (self.data_frame.DATE <= out_date)]
        stations_w_complete_records_ = df_w_quanlified_dates.groupby(self.unique_column_name).count()[self.response_var] == self.month_count
        stations_w_complete_records = list(stations_w_complete_records_.index[stations_w_complete_records_.values])
        self.output = df_w_quanlified_dates[df_w_quanlified_dates[self.unique_column_name].isin(stations_w_complete_records)]

        return self

    def add_sst_info(self, nc_data_dir = None, nc_mask_dir = None):

        if nc_mask_dir is None:
            nc_mask_dir = os.path.join(os.getcwd(), 'data/lsmask.nc')
        if nc_data_dir is None:
            nc_data_dir = os.path.join(os.getcwd(), 'data/sst.mnmean.nc')

        begining_month = min(self.output.DATE).month
        begining_year = min(self.output.DATE).year

        ending_month = max(self.output.DATE).month
        ending_year = max(self.output.DATE).year

        begin_string = '-'.join([str(begining_year), str(begining_month).zfill(2)])
        ending_string = '-'.join([str(ending_year), str(ending_month).zfill(2)])

        cal = SSTcalculator(nc_data_dir = nc_data_dir, nc_mask_dir = nc_mask_dir)
        lats_and_lons = self.output.groupby(self.unique_column_name).first().reset_index()[[self.unique_column_name, 'LONGITUDE', 'LATITUDE']]

        list_of_forms = cal.look_up_engine(starting_date = begin_string, end_date= ending_string, spatial_points = lats_and_lons)
        sst_forms = pd.concat(list_of_forms, axis = 0) #stack up (by row)
        sst_forms = sst_forms.drop(['LONGITUDE','LATITUDE'], axis =1) #drop a column

        temp1 = self.output.set_index([self.unique_column_name, 'YEAR', 'MONTH']).copy()
        temp2 = sst_forms.set_index([self.unique_column_name, 'YEAR', 'MONTH']).copy()

        self.output = pd.merge(temp1, temp2, left_index= True, right_index= True).reset_index()

        return self

    def flip_to_flat_and_wide(self, vars_to_retain):

        if vars_to_retain is None:
            vars_to_retain = ['PRCP', 'SST','RANGE_HIGH','RANGE_OVERALL', 'RANGE_LOW','TMIN', 'TMAX']

        column_name = []

        for var in vars_to_retain:
            for i in range(self.month_count):
                column_name.append(var + str(i + 1))


        location_info = self.output[[self.unique_column_name, 'LATITUDE', 'LONGITUDE']]
        df_reindexed = self.output.set_index([self.unique_column_name, 'DATE'])

        unique_locations = location_info.groupby(self.unique_column_name).first().reset_index().copy()

        temp1 = df_reindexed.unstack()[vars_to_retain].values
        temp1_index = df_reindexed.unstack()[vars_to_retain].index

        temp1_reset_index = pd.DataFrame(temp1, columns = column_name, index = temp1_index).reset_index()
        temp1_reset_index.set_index(self.unique_column_name, inplace = True)
        unique_locations.set_index(self.unique_column_name, inplace = True)

        self.output = temp1_reset_index.merge(unique_locations, left_index = True, right_index = True).reset_index()
        return self

    def get_values(self):

        try:
            self.output.drop(['Unnamed: 0'], axis = 1, inplace = True)
        except KeyError:
            pass

        return self.output

if __name__ == '__main__':

    # ---- working with rainfall data ------


    target_dir_rain = './data/monthly_rainfall.csv'
    monthly_rain = pd.read_csv(target_dir_rain)

    one_year_rain = PreprocessingPipeline(monthly_rain, unique_column_name = 'STATION', response_var = 'PRCP').\
    select_given_range(in_date = '2015-01',out_date = '2015-12').\
    add_sst_info().\
    get_values()

    five_year_rain = PreprocessingPipeline(monthly_rain, unique_column_name = 'STATION', response_var = 'PRCP').\
    select_given_range(in_date = '2011-01', out_date = '2015-12').\
    add_sst_info().\
    get_values()

    one_year_rain_fw = PreprocessingPipeline(monthly_rain, unique_column_name = 'STATION', response_var = 'PRCP').\
    select_given_range(in_date = '2015-01', out_date = '2015-12').\
    add_sst_info().\
    flip_to_flat_and_wide(vars_to_retain = ['PRCP', 'TMAX','TMIN', 'SST']).\
    get_values()

    five_year_rain_fw = PreprocessingPipeline(monthly_rain, unique_column_name = 'STATION', response_var = 'PRCP').\
    select_given_range(in_date = '2011-01', out_date = '2015-12').\
    add_sst_info().\
    flip_to_flat_and_wide(vars_to_retain = ['PRCP', 'TMAX','TMIN', 'SST']).\
    get_values()

    # ---- working with flood data ------

    target_dir_flood = './data/flood_data_5_years.csv'
    monthly_flood = pd.read_csv(target_dir_flood)

    one_year_flood = PreprocessingPipeline(monthly_flood, unique_column_name = 'SITENUMBER', response_var = 'GAGE_MAX').\
    select_given_range(in_date = '2015-01',out_date = '2015-12').\
    add_sst_info().\
    get_values()

    five_year_flood  = PreprocessingPipeline(monthly_flood, unique_column_name = 'SITENUMBER', response_var = 'GAGE_MAX').\
    select_given_range(in_date = '2011-01', out_date = '2015-12').\
    add_sst_info().\
    get_values()

    one_year_rain_fw = PreprocessingPipeline(monthly_flood, unique_column_name = 'SITENUMBER', response_var = 'GAGE_MAX').\
    select_given_range(in_date = '2015-01', out_date = '2015-12').\
    add_sst_info().\
    flip_to_flat_and_wide(vars_to_retain = ['GAGE_MAX']).\
    get_values()

    five_year_rain_fw = PreprocessingPipeline(monthly_flood, unique_column_name = 'SITENUMBER', response_var = 'GAGE_MAX').\
    select_given_range(in_date = '2011-01', out_date = '2015-12').\
    add_sst_info().\
    flip_to_flat_and_wide(vars_to_retain = ['GAGE_MAX']).\
    get_values()


