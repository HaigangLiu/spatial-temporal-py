import pymc3 as pm
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt
from theano import scan

from pymc3.distributions import continuous
from pymc3.distributions import distribution

class ArealModelPreprocessor:
    '''
    Calculate necessary quantities for the areal model
    i.e.: adjacent matrix and weight matrix.

    Args:
        spatial_data_frame(pandas dataframe): a data frame with spatial information (LATITUDE and LONGITUDE columns)
        key (string): the name of the column that contains unique identifier for each location
        watershed (string): the name of the column that contains the watershed information
        because same watershed locations are considered as neighbors.

    '''
    def __init__(self, spatial_data_frame, key='SITENUMBER', watershed='WATERSHED'):
        self.spatial_data_frame = spatial_data_frame
        self.key = key
        self.watershed = watershed

    def generate_adjacent_matrix(self):
        '''
        find the adjacent matrix and weight matrix for each location
        based on the dummy variable of region ('WATERSHED')
        returns:
            adjacent_matrix, index for neighbors. [2, 4] means the touch border with 3rd and 5th observation
            weight_matrix: replace all elements in adjacent matrix with 1.
        '''
        watershed_list = self.spatial_data_frame['WATERSHED']
        neighbor_matrix = [] # an n by n matrix, 1 for adjcent, 0 for not.
        adjacent_matrix = []
        weight_matrix = []

        for entry in watershed_list:
            w = [0 if entry!=comp else 1 for comp in watershed_list]
            neighbor_matrix.append(w)
        neighbor_matrix = np.array(neighbor_matrix)

        for idx, row in enumerate(neighbor_matrix):
            mask = np.argwhere(row == 1).ravel().tolist()
            mask.remove(idx) #delete the location itself.

            adjacent_matrix.append(mask)
            weight_matrix.append([1]*len(mask))

        return adjacent_matrix, weight_matrix

    def pad_matrix(self, input_list):
        '''
        construct the matrix of W by making sure that every
        row has the same length.
        args:
            input list (list): a list of spatial information for each location
        return:
            the padded matrix with rows of equal length.
        '''
        max_vector_length = max([len(row) for row in input_list])
        input_list_copy = input_list.copy()

        for row in input_list_copy:
            while len(row) < max_vector_length:
                row.append(0)
        return np.array(input_list_copy)

    def run(self):

        adjacent_matrix_, weight_matrix_ = self.generate_adjacent_matrix()
        adjacent_matrix = self.pad_matrix(adjacent_matrix_)
        weight_matrix = self.pad_matrix(weight_matrix_)
        return adjacent_matrix, weight_matrix

class CAR(distribution.Continuous):
    def __init__(self, w, a, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 0.
        self.a = a = tt.as_tensor_variable(a)
        self.w = w = tt.as_tensor_variable(w)
        self.tau = tau*tt.sum(w, axis=1)

    def compute_mu_elementwise(self, x):
        mu_w, _ = scan(fn= lambda w, a: tt.sum(w*x[a])/tt.sum(w), sequences=[self.w, self.a])
        return mu_w

    def logp(self, x):
        mu_w = self.compute_mu_elementwise(x)
        return tt.sum(continuous.Normal.dist(mu=mu_w, tau=self.tau).logp(x))

class ArealModel:
    def __init__(self, y, x, adjacent_matrix, weight_matrix):
        self.y = y
        self.x = x
        self.N = len(y)
        self.adjacent_matrix = adjacent_matrix
        self.weight_matrix = weight_matrix

    def fit(self):
        self.model_car = pm.Model()
        with self.model_car:
            # beta_0 = pm.Normal('beta_0', mu = 0, tau = 10e-5)
            # beta_1 = pm.Normal('beta_1', mu = 0, tau = 10e-5)

            # tau_h = pm.Gamma('tau_h', alpha=1, beta=1.0)
            tau_c = pm.Gamma('tau_c', alpha=1.0, beta=1.0)
            # sd = pm.Gamma('tau_d', alpha=1.0, beta=1.0)

            theta = pm.Normal('theta', mu=0.0, sd=3, shape=1)
            mu_phi = CAR('mu_phi', w=self.weight_matrix, a=self.adjacent_matrix, tau=tau_c, shape=self.N)

            # mu_total = pm.Deterministic('mu', )

            Yi = pm.Normal('Yi', mu=mu_phi + theta*x, sd=3,  observed=y)

            trace = pm.sample(5000, cores=2, tune=1000)

        return trace

if __name__ == '__main__':

    temp_data = pd.read_csv('./rainfall_data_nc/merged_20151003.csv')
    temp_data = temp_data[~np.isnan(temp_data.GAGE_MAX)]
    temp_data = temp_data[~np.isnan(temp_data.PRCP)]

    weight_matrix, adjacent_matrix = ArealModelPreprocessor(temp_data, key='SITENUMBER').run()

    x = temp_data['PRCP'].values
    y = temp_data['GAGE_MAX'].values

    model = ArealModel(y, x, adjacent_matrix, weight_matrix)
    trace = model.fit()



    # dataframe = areal_data.data
    # adjacent_matrix = areal_data.adjacent_matrix
    # weight_matrix = areal_data.weight_matrix
    # padded_weight_matrix = areal_data.padded_weight_matrix
    # padded_adjacent_matrix = areal_data.padded_adjacent_matrix
    # x = dataframe['PRCP'].values
    # y = dataframe['GAGE_MAX'].values
    # N = len(x)


# class CAR(distribution.Continuous):

#     def __init__(self, w, a, tau, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.mode = 0.
#         self.a = a = tt.as_tensor_variable(a)
#         self.w = w = tt.as_tensor_variable(w)
#         self.tau = tau*tt.sum(w, axis=1)

#     def compute_mu_elementwise(self, x):
#         mu_w, _ = scan(fn= lambda w, a: tt.sum(w*x[a])/tt.sum(w), sequences=[self.w, self.a])
#         return mu_w

#     def logp(self, x):
#         mu_w = self.compute_mu_elementwise(x)
#         return tt.sum(continuous.Normal.dist(mu=mu_w, tau=self.tau).logp(x))


# model_car = pm.Model()

# with model_car:
#     # beta_0 = pm.Normal('beta_0', mu = 0, tau = 10e-5)
#     # beta_1 = pm.Normal('beta_1', mu = 0, tau = 10e-5)

#     # tau_h = pm.Gamma('tau_h', alpha=1, beta=1.0)
#     tau_c = pm.Gamma('tau_c', alpha=1.0, beta=1.0)
#     # sd = pm.Gamma('tau_d', alpha=1.0, beta=1.0)

#     theta = pm.Normal('theta', mu=0.0, sd=3, shape=N)
#     mu_phi = CAR('mu_phi', w=padded_weight_matrix, a=padded_adjacent_matrix, tau=tau_c, shape=N)

#     # mu_total = pm.Deterministic('mu', )

#     Yi = pm.Normal('Yi', mu=mu_phi + theta*x, sd=3,  observed=y)

#     trace = pm.sample(5000, cores=2, tune=1000)


