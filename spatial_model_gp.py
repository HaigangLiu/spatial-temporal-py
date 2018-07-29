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
    def __init__(self, data_frame, response_var, convert_coordinates = True):

        self.train_loc_cache = data_frame[['LATITUDE','LONGITUDE']]
        self.convert_coordinates = convert_coordinates

        if convert_coordinates:
            coordinates = coordinates_converter(data_frame)
        else:
            coordinates = self.train_loc_cache
            print('Lat and lon used directly to calculate Euclidean distance')

        self.response_var = response_var
        self.X = shared(coordinates.values)
        self.y = np.log(data_frame[self.response_var].values)

    def build_gp_model(self, sampling_size = 5000, trace_plot_name = None):
        '''
        Args:
            sampling_size (int): the length of markov chain
            create_traceplot (boolean): Whether or not generate the traceplot.
        '''
        with pm.Model() as model:
            rho = pm.Exponential('rho', 1/5, shape = 2)
            cov_func = pm.gp.cov.Matern52(2, ls = rho)
            mean_prior = pm.Exponential('mean_prior', 1/3)
            c = pm.Normal('constant_mean', mu = mean_prior, sd = 4,  shape = 98)

            gp = pm.gp.Marginal(mean_func = pm.gp.mean.Constant(c), cov_func = cov_func)
            sigma = pm.HalfNormal("sigma", sd = 3)
            y_ = gp.marginal_likelihood("y",
                                        X = self.X,
                                        y = self.y,
                                        noise = sigma)
            start = pm.find_MAP()
            self.trace = pm.sample(sampling_size, nchains = 1)

        self.model = model

        if trace_plot_name:
            fig, axs = plt.subplots(3, 2) # 2 RVs
            pm.traceplot(self.trace, varnames = ['rho', 'sigma', 'constant_mean'], ax = axs)
            fig.savefig(trace_plot_name)
            fig_path = os.path.join(os.getcwd(), trace_plot_name)
            print(f'the traceplot has been saved to {fig_path}')

    def predict(self, new_data_frame):
        '''
        Args:
            new_data_frame (pandas dataframe): the dataframe of new locations. Users can also include the truth value of Y.
            Note that MSE cannot be computed if truth is not provided.
        '''
        self.test_loc_cache = new_data_frame[['LATITUDE', 'LONGITUDE']]

        if self.convert_coordinates:
            X_new = coordinates_converter(new_data_frame).values
        else:
            X_new = self.test_loc_cache
            print('Lat and lon used directly to calculate Euclidean distance')

        self.X.set_value(X_new)
        with self.model:
            self.predicted_values = pm.sample_ppc(self.trace)

        median = np.median(self.predicted_values['y'], axis = 0)
        try:
            self.Y_new = new_data_frame[self.response_var].values
        except:
            print('truth column not provided, thus metrics like MSE are not calculated.')
        else:
            predicted_vals_transformed_back = np.exp(median)
            l1_loss = np.mean(np.abs(predicted_vals_transformed_back - self.Y_new))
            l2_loss = np.mean(np.square(predicted_vals_transformed_back - self.Y_new))
            self.summary = {'l1_loss': l1_loss, 'l2_loss': l2_loss}
        return median

if __name__ == '__main__':

    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')
    data_list = list(range(len(data)))
    test_list =  np.random.choice(data_list, 0,  replace = False)
    train_list = [i for i in data_list if i not in test_list]

    test_case = GPModelSpatial(data.iloc[train_list], 'PRCP', convert_coordinates = True)
    test_case.build_gp_model(trace_plot_name = 'test_traceplot.png')
    vars_ = test_case.predict(data.iloc[train_list])
    #vars_ = test_case.predict(data.iloc[test_list])

    import pickle
    with open('result.pickle', 'wb') as handler:
        pickle.dump(test_case, handler, protocol=pickle.HIGHEST_PROTOCOL)
    print(test_case.summary)
