import warnings
warnings.filterwarnings('ignore')

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utilities_functions import coordinates_converter

class GPModelSpatialSparse:
    '''
    Build a spatial temporal model based on Pymc3 framework.
    Args:
        df (pandas dataframe): a data frame LATITUDE and LONGITUDE and reponse variable
        response_var (str): The name of column of Y
        split_ratio(float): The ratio to split training set and test set.
            NOTE: set it to 1 if user only wants to train on the whole set.
    '''
    def __init__(self, df, response_var = 'PRCP', spilt_ratio = 0.7):

        X = coordinates_converter(df).values
        self.response_var = response_var
        y = df[self.response_var].values

        all_index = list(range(len(df)))
        train_size  = int(round(len(df)*spilt_ratio,0))

        train_index = np.random.choice(all_index, train_size)
        test_index = [idx for idx in all_index if idx not in train_index]

        self.X_train = X[train_index]; self.X_test = X[test_index]
        self.y_train = y[train_index]; self.y_test = y[test_index]

        self.train_loc_cache = df.loc[train_index, ['LATITUDE','LONGITUDE']]
        self.test_loc_cache = df.loc[test_index, ['LATITUDE','LONGITUDE']]

    def fit(self, size = 5000, nodes = 10, traceplot_name = None):
        '''
        Args:
            size (int): the length of markov chain
            create_traceplot (boolean): Whether or not generate the traceplot.
        '''
        self.model = pm.Model()
        with self.model:

            tau = pm.Gamma("tau", alpha=2, beta=1, shape=3)
            eta = pm.HalfCauchy("eta", beta=5)
            cov = eta**2 * pm.gp.cov.Matern52(3, tau)

            self.gp = pm.gp.MarginalSparse(cov_func = cov, approx="VFE")
            Xu = pm.gp.util.kmeans_inducing_points(nodes, self.X_train)

            sigma = pm.HalfNormal('sigma', sd = 4)
            y_ = self.gp.marginal_likelihood("y", X= self.X_train, Xu=Xu, y= np.log(self.y_train), noise= sigma)

            start = pm.find_MAP()
            self.trace = pm.sample(size, start = start )

        if traceplot_name:
            fig, axs = plt.subplots(3, 2) # 2 RVs
            pm.traceplot(self.trace, varnames = ['tau', 'eta', 'sigma'], ax = axs)
            fig.savefig(traceplot_name)
            fig_path = os.path.join(os.getcwd(), traceplot_name)
            print(f'the traceplot has been saved to {fig_path}')

    def predict(self, new_df = None, sample_size = 1000):
        '''
        Args:
            new_df (pandas dataframe): the dataframe of new locations. Users can also include the truth value of Y.
            Note that MSE cannot be computed if truth is not provided.
        '''
        if new_df:
            #allow user to pass new data later
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

        return output_df

if __name__ == '__main__':

    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')

    sparse_gp_model = GPModelSpatialSparse(data, 'PRCP')
    sparse_gp_model.fit(size = 20000, nodes = 15, traceplot_name = 'test_traceplot_sparse.png')
    output_df = sparse_gp_model.predict()

    import pickle
    with open('result_sparse.pickle', 'wb') as handler:
        pickle.dump(sparse_gp_model, handler, protocol=pickle.HIGHEST_PROTOCOL)
    print(sparse_gp_model.summary)
