import warnings
warnings.filterwarnings('ignore')
import pymc3 as pm
import numpy as np
import pandas as pd
from SSTcalculator import SSTcalculator
import matplotlib.pyplot as plt
from theano import shared
import os
class GPModelSpatial:

    '''
    Build a spatial temporal model based on Pymc3 framework.
    Args:
        data_frame (pandas dataframe): a data frame LATITUDE and LONGITUDE and reponse variable
        response_var (str): The name of column of Y
    '''
    def __init__(self, data_frame, response_var):

        d1, d2, d3 = SSTcalculator._lat_lon_to_cartesian(data_frame['LATITUDE'], data_frame['LONGITUDE'])

        self.response_var = response_var
        self.X = shared(np.array([d1, d2, d3]).T)
        self.y = np.log(data_frame[self.response_var].values)

    def build_gp_model(self, sampling_size = 5000, trace_plot_name = None):
        '''
        Args:
            sampling_size (int): the length of markov chain
            create_traceplot (boolean): Whether or not generate the traceplot.
        '''
        with pm.Model() as model:

            rho = pm.Exponential('rho', 1, shape = 3)
            cov_func = pm.gp.cov.Matern52(3, ls = rho)

            gp = pm.gp.Marginal(cov_func = cov_func)
            sigma = pm.HalfNormal("sigma", sd = 3)
            y_ = gp.marginal_likelihood("y",
                                        X = self.X,
                                        y = self.y,
                                        noise = sigma)
            start = pm.find_MAP()
            self.trace = pm.sample(sampling_size, nchains = 1)

        self.model = model

        if trace_plot_name:
            fig, axs = plt.subplots(2, 2) # 2 RVs
            pm.traceplot(self.trace, varnames = ['rho', 'sigma'], ax = axs)
            fig.savefig(trace_plot_name)
            fig_path = os.path.join(os.getcwd(), trace_plot_name)
            print(f'the traceplot has been saved to {fig_path}')

    def predict(self, new_data_frame):
        '''
        Args:
            new_data_frame (pandas dataframe): the dataframe of new locations. Users can also include the truth value of Y.
            Note that MSE cannot be computed if truth is not provided.
        '''
        x_new, y_new, z_new = SSTcalculator._lat_lon_to_cartesian(new_data_frame['LATITUDE'], new_data_frame['LONGITUDE'])
        X_new = np.array([x_new, y_new, z_new]).T
        self.X.set_value(X_new)

        with self.model:
            self.predicted_values_raw = pm.sample_ppc(self.trace)

        median_of_predicted_values = np.median(self.predicted_values_raw['y'], axis = 0)
        try:
            Y_new = new_data_frame[self.response_var].values
        except:
            print('truth column not provided, thus metrics like MSE are not calculated.')
        else:
            predicted_vals_transformed_back = np.exp(median_of_predicted_values)
            l1_loss = np.mean(np.abs(predicted_vals_transformed_back - Y_new))
            l2_loss = np.mean(np.square(predicted_vals_transformed_back - Y_new))
            self.summary = {'l1_loss': l1_loss, 'l2_loss': l2_loss}
        return median_of_predicted_values


if __name__ == '__main__':

    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')

    data_list = list(range(len(data)))
    test_list =  np.random.choice(data_list, 30,  replace = False)
    train_list = [i for i in data_list if i not in test_list]

    test_case = GPModelSpatial(data.iloc[train_list], 'PRCP')
    test_case.build_gp_model(trace_plot_name = 'test_traceplot.png')
    vars_ = test_case.predict(data.iloc[test_list])

    print(test_case.summary)
      # self.predicted_val_medians = self.predicted_values_raw['y'].mean(axis = 0)
      #   self.predicted_val_975 = self.predicted_values_raw['y'].percentile(97.5, axis = 0)
      #   self.predicted_val_025 = self.predicted_values_raw['y'].percentile(2.5, axis = 0)
