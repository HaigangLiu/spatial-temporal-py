import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from fill_missing_dates import fill_missing_dates


class MissingDataHandler:

    def __init__(self, dataframe, spatial_column, temporal_column,
       variable_column):

        self.spatial_column = spatial_column
        self.temporal_column = temporal_column
        self.variable_column = variable_column
        self.report = None
        self.df = dataframe[[spatial_column,temporal_column,variable_column]]

    def get_missing_data_report(self):

        if self.report is None:

            self.df[self.temporal_column] = pd.to_datetime(self.df[self.temporal_column]) #set
            # value
            self.df = self.df.set_index([self.temporal_column, self.spatial_column]).unstack()

            yearly = self.df.resample('Y').count() #will count non-na automatically

            number_of_days = []; index_ =[]
            for i in yearly.index:
                if i.year%4 == 0:
                    number_of_days.append(366)
                    index_.append(str(i.year))
                else:
                    number_of_days.append(365)
                    index_.append(str(i.year))

            yearly.index = index_
            self.report = (yearly[self.variable_column].T/np.array
                (number_of_days)).reset_index()

        return self.report

    def get_missing_data_table(self, threshold):

        if self.report is None:
            _ = self.get_missing_data_report()

        def coloring_scheme(cell_value):
            if cell_value > threshold:
                color = 'green'
            else:
                color = 'red'
            return 'background-color: %s' % color

        colored_cell = self.report.style.applymap(coloring_scheme, subset=[col
           for
            col in self.report.columns if col!=self.spatial_column])
        dir_ = os.path.join(os.getcwd(), 'data_missingess.xlsx')
        colored_cell.to_excel(dir_)
        print(f'a colored excel file has been saved to {dir_}')
        return colored_cell

    def get_missing_data_tsplot(self, thresholds=[0.85, 0.88, 0.9, 0.95, 0.975, 0.99]):

        if self.report is None:
            _ = self.get_missing_data_report()

        available_sites = {}

        for threshold_ in thresholds:
            year_columns = list(self.report.columns)

            while year_columns:
                completed_years = np.sum(self.report[year_columns] > threshold_, axis=1)
                bool_col = completed_years == len(year_columns)
                available_sites[str(len(year_columns))] = self.report.loc[bool_col,self.spatial_column].values
                year_columns.pop(0)

            ts_plot_x = []; ts_plot_y = []
            for k, v in available_sites.items():
                ts_plot_y.append(len(v)); ts_plot_x.append(int(k))
            plt.plot(ts_plot_x, ts_plot_y, marker='o', alpha=0.6)

        plt.title('available stations over 10 years')
        plt.legend([0.85, 0.9, 0.95, 0.975, 0.99])
        dir_fig = os.path.join(os.getcwd(), 'data_missingess_.png')
        plt.savefig(dir_fig, dpi=400)
        plt.close()

        print(f'a time series plot of number of avalaible stations change overtime has been saved to {dir_fig}')
        return available_sites

if __name__ == '__main__':
    df = pd.read_csv('./data/with_height.csv', dtype ={'SITENUMBER':str}, index_col=0 ).reset_index(drop=True)
    df_2 = fill_missing_dates(df, 'SITENUMBER', 'DATE', ['LATITUDE', 'LONGITUDE', 'MEDIAN_HISTORICAL', 'ELEVATION'])

    hanlder = MissingDataHandler(df_2, 'SITENUMBER', 'DATE', 'GAGE_MAX')
    plot_ = hanlder.get_missing_data_tsplot()
    table_ = hanlder.get_missing_data_table(threshold=0.9)
    report_ = hanlder.get_missing_data_report()




