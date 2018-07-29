import warnings
warnings.filterwarnings('ignore')
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from theano import shared
import os
from utilities_functions import coordinates_converter

class GPModelSpatial:
    '''
    Build a spatial temporal model based on Pymc3 framework.
    Args:
        data_frame (pandas dataframe): a data frame LATITUDE and LONGITUDE and reponse variable
        response_var (str): The name of column of Y
    '''
    def __init__(self, data_frame, response_var, covert_coordinates = True):

        if covert_coordinates:
            new_coordinates = coordinates_converter(data_frame)
        else:
            new_coordinates = data_frame[['LATITUDE','LONGITUDE']]

        self.response_var = response_var
        self.X = shared(new_coordinates.values)
        self.y = np.log(data_frame[self.response_var].values)
        self.train_lat_lon = data_frame[['LATITUDE','LONGITUDE']]

    def build_gp_model(self, sampling_size = 5000, trace_plot_name = None):
        '''
        Args:
            sampling_size (int): the length of markov chain
            create_traceplot (boolean): Whether or not generate the traceplot.
        '''
        with pm.Model() as model:
            rho = pm.Exponential('rho', 0.05, shape = 3)
            cov_func = 4*pm.gp.cov.Matern52(3, ls = rho)

            gp = pm.gp.Marginal(mean_func = pm.gp.mean.Constant(), cov_func = cov_func)
            sigma = pm.HalfNormal("sigma", sd = 2)
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
        X_new = coordinates_converter(new_data_frame).values
        self.X.set_value(X_new)
        self.test_lat_lon = new_data_frame[['LATITUDE', 'LONGITUDE']]

        with self.model:
            self.predicted_values_raw = pm.sample_ppc(self.trace)

        median_of_predicted_values = np.median(self.predicted_values_raw['y'], axis = 0)
        try:
            self.Y_new = new_data_frame[self.response_var].values
        except:
            print('truth column not provided, thus metrics like MSE are not calculated.')
        else:
            predicted_vals_transformed_back = np.exp(median_of_predicted_values)
            l1_loss = np.mean(np.abs(predicted_vals_transformed_back - self.Y_new))
            l2_loss = np.mean(np.square(predicted_vals_transformed_back - self.Y_new))
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

    import pickle
    with open('result.pickle', 'wb') as handler:
        pickle.dump(test_case, handler, protocol=pickle.HIGHEST_PROTOCOL)
    print(test_case.summary)
