import warnings
warnings.filterwarnings('ignore')
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utilities_functions import coordinates_converter

class GPModelSpatial:
    '''
    Build a spatial temporal model based on Pymc3 framework.
    Args:
        data_frame (pandas dataframe): a data frame LATITUDE and LONGITUDE and reponse variable
        response_var (str): The name of column of Y
    '''
    def __init__(self, df, response_var  = 'PRCP', split_ratio= 0.7):

        X = coordinates_converter(df).values
        self.response_var = response_var
        y = df[self.response_var].values

        all_index = list(range(len(df)))
        train_size  = int(round(len(df)*split_ratio,0))

        train_index = np.random.choice(all_index, train_size)
        test_index = [idx for idx in all_index if idx not in train_index]

        self.X_train = X[train_index]; self.X_test = X[test_index]
        self.y_train = y[train_index]; self.y_test = y[test_index]

        self.train_loc_cache = df.loc[train_index, ['LATITUDE','LONGITUDE']]
        self.test_loc_cache = df.loc[test_index, ['LATITUDE','LONGITUDE']]

    def fit(self, sampling_size = 5000, traceplot_name = None, fast_sampling = False):
        '''
        Args:
            sampling_size (int): the length of markov chain
            create_traceplot (boolean): Whether or not generate the traceplot.
        '''
        self.model = pm.Model()
        with self.model:
            rho = pm.Exponential('rho', 1/5, shape = 3)
            tau = pm.Exponential('tau', 1/3)

            cov_func = pm.gp.cov.Matern52(3, ls = rho)
            self.gp = pm.gp.Marginal(cov_func = cov_func)

            sigma = pm.HalfNormal('sigma', sd = 3)
            y_ = self.gp.marginal_likelihood('y',
                                        X = self.X_train,
                                        y = np.log(self.y_train),
                                        noise = sigma)

        if fast_sampling:
            with self.model:
                inference = pm.ADVI()
                approx = pm.fit(n = 50000, method=inference) #until converge
                self.trace = approx.sample(draws = sampling_size)

        else:
            with self.model:
                start = pm.find_MAP()
                self.trace = pm.sample(sampling_size, nchains = 1)

        if traceplot_name:
            fig, axs = plt.subplots(3, 2) # 2 RVs
            pm.traceplot(self.trace, varnames = ['rho', 'sigma', 'tau'], ax = axs)
            fig.savefig(traceplot_name)
            fig_path = os.path.join(os.getcwd(), traceplot_name)
            print(f'the traceplot has been saved to {fig_path}')

    def predict(self, new_df = None, sample_size = 1000):
        '''
        Args:
            new_data_frame (pandas dataframe): the dataframe of new locations. Users can also include the truth value of Y.
            Note that MSE cannot be computed if truth is not provided.
        '''
        if new_df:
            try:
                self.X_test = coordinates_converter(new_df)
                self.y_test = new_df[self.response_var]
                self.test_loc_cache = new_df[['LATITUDE','LONGITUDE']]
            except:
                raise ValueError('The new dataframe should contain LATITUDE, LONGITUDE and the variable column, e.g., PRCP')

        with self.model:
            y_pred = self.gp.conditional("y_pred", self.X_test)
            self.simulated_values = pm.sample_ppc(self.trace, vars=[y_pred], samples= sample_size)
            self.predictions = np.exp(np.median(self.simulated_values['y_pred'], axis = 0))

        l1_loss = np.mean(np.abs(self.predictions - self.y_test))
        l2_loss = np.mean(np.square(self.predictions - self.y_test))
        self.summary = {'l1_loss': l1_loss, 'l2_loss': l2_loss}

        output_df = self.test_loc_cache.copy()
        output_df['PRED'] = self.predictions

        return self.predictions

if __name__ == '__main__':
    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')

    gp_spatial_model = GPModelSpatial(data, split_ratio = 0.7, response_var = 'PRCP')
    gp_spatial_model.fit(traceplot_name = 'test_traceplot.png', fast_sampling = True)
    vars_ = gp_spatial_model.predict()

    import pickle
    with open('result.pickle', 'wb') as handler:
        pickle.dump(gp_spatial_model, handler, protocol=pickle.HIGHEST_PROTOCOL)

    print(gp_spatial_model.summary)
    print(gp_spatial_model.y_test)
    print(gp_spatial_model.predictions)
    #{'l1_loss': 2.4585309225114274, 'l2_loss': 11.003350489559711}
