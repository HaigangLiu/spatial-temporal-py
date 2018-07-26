import numpy as np
from sklearn import linear_model
import pandas as pd

class SimpleTrendRemover:
    '''
    Find the average the trend of Y (rainfall volume),
    and remove it and return a new data frame. A simple ordinary square method is used to implement this harmonic regression.

    Args:
        data_frame (pandas dataframe): dataframe with monthly data
        flat_and_wide (boolean): must specify this
        becasue these two situations are handled very differently
        response_var (string): name of the response variable
        periods (int): harmonic regression requires cycle length parameter.
    '''

    def __init__(self, data_frame, flat_and_wide, response_var = 'PRCP', periods = 12):

        self.data_frame = data_frame.copy()
        self.flat_and_wide = flat_and_wide
        self.response_var = response_var
        self.periods = periods
        self._find_the_trend()

    def _find_the_trend(self):

        if self.flat_and_wide:

            avg = self.data_frame.mean(axis = 0)
            self._response_vars = []
            for i in avg.index:
                if i.startswith(self.response_var):
                    self._response_vars.append(i)
            Y = avg[self._response_vars].values

        else:
            Y = self.data_frame.groupby(['YEAR','MONTH']).mean()[self.response_var].values

        timeline = np.arange(1, 1 + len(Y))
        cos_term = np.cos(timeline*np.pi/(self.periods/2.0))
        sin_term = np.sin(timeline*np.pi/(self.periods/2.0))
        X = np.array([cos_term, sin_term]).T

        regression = linear_model.LinearRegression(fit_intercept = True)
        regression.fit(X, Y)
        cosine_coef, sin_coef = regression.coef_
        intercept = regression.intercept_

        self.avg_trend = regression.predict(X)

    def update_dataframe_with_resid(self):

        if self.flat_and_wide:
            self.data_frame[self._response_vars] = self.data_frame[self._response_vars] - self.avg_trend
        else:
            number_of_stations = len(np.unique(self.data_frame['STATION']))
            trends_duplicated_by_station = np.tile(self.avg_trend, number_of_stations)
            self.data_frame[self.response_var] = self.data_frame[self.response_var] - trends_duplicated_by_station

        return self.data_frame

def add_seasonal_indicator(dataframe, new_col = 'SEASON', rule = None):

    dataframe[new_col] = 'Fall'

    winter_mask = dataframe.MONTH.isin([12, 1, 2])
    spring_mask = dataframe.MONTH.isin([3, 4, 5])
    summer_mask = dataframe.MONTH.isin([6, 7, 8])

    dataframe[new_col][winter_mask]= 'Winter'
    dataframe[new_col][spring_mask]= 'Spring'
    dataframe[new_col][summer_mask] = 'Summer'

    indicators = pd.get_dummies(dataframe[new_col], drop_first= True)

    return pd.concat([dataframe, indicators], axis = 1)


if __name__ == '__main__':

    df = pd.read_csv('/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years_flat_and_wide.csv')
    trend_remover = SimpleTrendRemover(data_frame= df, response_var='PRCP', flat_and_wide= True)
    f = trend_remover.update_dataframe_with_resid()
    f.to_csv('/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years_flat_and_wide_residual.csv')

    # df2 = pd.read_csv('/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years.csv')
    trend_remover2 = SimpleTrendRemover(data_frame= df2, response_var='PRCP', flat_and_wide= False)
    f2 = trend_remover2.update_dataframe_with_resid()
    # f2.to_csv('/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years_residual.csv')

