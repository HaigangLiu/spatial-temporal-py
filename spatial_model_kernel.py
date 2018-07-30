import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sklearn.cluster import KMeans
from theano import shared
from utilities_functions import coordinates_converter

class SpatialKernelModel:

    '''
    This model is inspired by Stroud et al (2001).
    The idea is
    1. place arbitrary number of 2d normal kernels on the spatial surface
    2. Assuming the observation is weighted average of these kernel densitites.
    3. Optimize the parameters that define the kernel.

    Note: the kernels will be selected based on KNN algorithm

    Args:
        data_frame (pandas dataframe): a data frame LATITUDE and LONGITUDE and reponse variable
        response_var (str): The name of column of Y
        number_of_centers (int): The number of spatial kernels
        split_ratio (float): The ratio to split training and test data
    '''
    def __init__(self, df, response_var = 'PRCP',  number_of_centers = 5, split_ratio = 0.7):

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

        self.number_of_centers = number_of_centers
        self.centers = self._auto_find_center()

    def _auto_find_center(self):
        kmeans = KMeans(n_clusters = self.number_of_centers, random_state=0).fit(self.X_train)
        centers = kmeans.cluster_centers_
        return pd.DataFrame(centers, columns = list('xyz'))

    def fit(self, sample_size, traceplot_name = None, fast_sampling = False):

        '''
        sample_size (int): The size of the sample
        traceplot_name (str): The name of the traceplot file
        fast_sampling   (bool): whether or not variational approximation should be used.

        Note: to evaluate the kernel function, pymc3 only accept tensor type from theano.
        '''
        self.model = pm.Model()
        # self.X_train = tt.constant(self.X_train) #need tensor type
        self.X_train = shared(self.X_train)

        with  self.model:
            evaluated_kernels = []
            packed_L = pm.LKJCholeskyCov('packed_L', n=3, eta=2., sd_dist = pm.HalfCauchy.dist(2.5))
            L = pm.expand_packed_triangular(3, packed_L)

            for center in self.centers.values:
                evaluated_kernels.append(pm.MvNormal.dist(mu = center, chol = L).logp(self.X_train))

            beta = pm.Normal('beta', mu = 0, sd = 3, shape = self.number_of_centers)
            latentProcess = pm.Deterministic('mu',tt.dot(beta, evaluated_kernels))

            error = pm.HalfCauchy('error', 12)
            y_ = pm.Normal("y", mu = latentProcess, sd = error, observed = np.log(self.y_train))

            if fast_sampling:
                with self.model:
                    inference = pm.ADVI()
                    approx = pm.fit(n = 50000, method=inference) #until converge
                    self.trace = approx.sample(draws = sample_size)

            else:
                with self.model:
                    start = pm.find_MAP()
                    self.trace = pm.sample(sample_size, start = start)

            if traceplot_name:
                fig, axs = plt.subplots(3, 2) # 2 RVs
                pm.traceplot(self.trace, varnames = ['packed_L', 'beta', 'error'], ax = axs)
                fig.savefig(traceplot_name)

                fig_path = os.path.join(os.getcwd(), traceplot_name)
                print(f'the traceplot has been saved to {fig_path}')

    def predict(self, new_df = None, sample_size = 500):

        if new_df:
            try:
                self.X_test = coordinates_converter(new_df)
                self.y_test = new_df[self.response_var]
                self.test_loc_cache = new_df[['LATITUDE','LONGITUDE']]
            except:
                raise ValueError('The new dataframe should contain LATITUDE, LONGITUDE and the variable column, e.g., PRCP')
        with self.model:
            self.X_train.set_value(self.X_test)
            self.simulated_values = pm.sample_ppc(self.trace, samples = sample_size)
            self.predictions = np.exp(np.median(self.simulated_values['y'], axis = 0))

        l1_loss = np.mean(np.abs(self.predictions - self.y_test))
        l2_loss = np.mean(np.square(self.predictions - self.y_test))

        self.summary = {'l1_loss': l1_loss, 'l2_loss': l2_loss}

        output_df = self.test_loc_cache.copy()
        output_df['PRED'] = self.predictions

        return self.predictions

if __name__ == '__main__':
    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')
    kernel_model = SpatialKernelModel(data, split_ratio = 0.7, response_var = 'PRCP', number_of_centers = 6)
    kernel_model.fit(20000, traceplot_name = 'kernel_method.png', fast_sampling = False)
    kernel_model.predict(sample_size = 10000)

    print(kernel_model.summary)
    print(kernel_model.y_test)
    print(kernel_model.predictions)
