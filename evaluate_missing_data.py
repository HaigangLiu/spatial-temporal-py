import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_preprocessing_tools import fill_missing_dates

class MissingSpatialTemporalDataHandler:
    '''
    Analyze the missingness of spatial temporal data in spatial and temporal dimension.
    get_missing_data_report() will give a table presenting spatial and temporal availablity
    get_missing_data_table() will color the aforementioned table within a threshold.
        e.g. a space-time combination with more than 90% data available will be marked as green. red otherwise
    get_missing_data_tsplot() will generate a time series graph of available
    station numbers over number of years one's looking at.

    example:
    >>> hanlder = MissingDataHandler(df_2, 'SITENUMBER', 'DATE', 'GAGE_MAX')
    >>> plot_ = hanlder.get_missing_data_tsplot()
    >>> table_ = hanlder.get_missing_data_table(threshold=0.9)
    >>> report_ = hanlder.get_missing_data_report()

    Args:
        dataframe(pandas dataframe): the dataset
        spatial_column(string): the name of the column of station names
        temporal_column(string): the name of the column of dates
        varname(string): the variable name (PRCP, GAGE or etc.)
    '''
    def __init__(self, dataframe, spatial_column, temporal_column, varname):

        self.spatial_column = spatial_column
        self.temporal_column = temporal_column
        self.varname = varname
        self.report = None
        self.dataframe = dataframe
        self.report = self.get_missing_data_report()

    def get_missing_data_report(self):
        if self.report is None:
            dataframe = self.dataframe.copy() #don't manipulate original!
            dataframe[self.temporal_column] = pd.to_datetime(dataframe[self.temporal_column]) #set value
            dataframe = dataframe.set_index([self.temporal_column, self.spatial_column]).unstack()
            yearly_non_missing_count = dataframe.resample('Y').count() #will count non-na automatically

            number_of_days = []; index_ =[]
            for i in yearly_non_missing_count.index:
                if i.year%4 == 0:
                    number_of_days.append(366)
                    index_.append(str(i.year))
                else:
                    number_of_days.append(365)
                    index_.append(str(i.year))
            yearly_non_missing_count.index = index_
            self.report = (yearly_non_missing_count[self.varname].T/np.array(number_of_days)).reset_index()
        return self.report

    def get_missing_data_table(self, threshold):

        def coloring_scheme(cell_value):
            if cell_value > threshold:
                color = 'green'
            else:
                color = 'red'
            return 'background-color: %s' % color

        colored_cell = self.report.style.applymap(coloring_scheme, subset=[col for col in self.report.columns if col!=self.spatial_column])
        dir_ = os.path.join(os.getcwd(), 'data_missingess.xlsx')
        colored_cell.to_excel(dir_)
        print(f'a colored excel file indicating the data completeness has been saved to {dir_}')
        return colored_cell

    def get_qualified_stations(self, years=7, threshold=0.9):

        if type(years) == int and years < 11:
            print(f'Data of recent {years} years from 2016 will be examined')
            years = [2016 -i for i in range(years)]
        elif years >= 11:
            raise IndexError('The maximum year allowed to look back is 10.')
            return None
        elif type(years) == list:
            pass
        else:
            raise TypeError('only accept list or int')
            return None

        years = [str(year) for year in years]
        report = self.report.copy()
        report.set_index(self.spatial_column, inplace=True)
        data_in_range = report[years]

        keeper = data_in_range[data_in_range > threshold]
        keeper.dropna(axis=0, inplace=True)
        return keeper.index.tolist()

    def get_dataframe(self, years=7, threshold=0.9):
        station_list = self.get_qualified_stations(years=years, threshold=threshold)
        df = self.dataframe.copy()

        mask = df[self.spatial_column].isin(station_list)
        selected_stations = df[mask]
        return selected_stations

    def get_missing_data_tsplot(self, thresholds=[0.85, 0.88, 0.9, 0.95, 0.975, 0.99], plotname=None):

        for threshold in thresholds:
            ts_plot_x = []; ts_plot_y = []
            for year in range(10):
                ts_plot_x.append(year+1)
                number_of_locs = self.get_qualified_stations(years=year+1, threshold=threshold)
                ts_plot_y.append(len(number_of_locs))

            plt.plot(ts_plot_x, ts_plot_y, marker='o', alpha=0.6)

        plotname = 'missing_data_evaluation.png' if plotname is None else plotname
        plt.title('Available stations counts over 10 years')
        plt.legend([0.85, .88, 0.9, 0.95, 0.975, 0.99])
        fig_dir = os.path.join(os.getcwd(), plotname)
        plt.savefig(fig_dir, dpi=250)
        plt.close()

        print(f'a time series plot of number of avalaible stations change overtime has been saved to {fig_dir}')

if __name__ == '__main__':
    df = pd.read_csv('./data/with_height.csv', dtype ={'SITENUMBER':str}, index_col=0 ).reset_index(drop=True)
    df_2 = fill_missing_dates(df, 'SITENUMBER', 'DATE', ['LATITUDE', 'LONGITUDE', 'MEDIAN_HISTORICAL', 'ELEVATION'])

    handler = MissingSpatialTemporalDataHandler(df_2, 'SITENUMBER', 'DATE',
       'GAGE_MAX')
    plot_ = handler.get_missing_data_tsplot()
    table_ = handler.get_missing_data_table(threshold=0.95)
    report_ = handler.get_missing_data_report()
    output = handler.get_qualified_stations(years=5, threshold=0.9)
    output_df = handler.get_dataframe(years=7, threshold=0.9)

    print(output_df.head())
