import pymc3 as pm
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt
from theano import scan

from pymc3.distributions import continuous
from pymc3.distributions import distribution

class ArealModel:
    def __init__(self, y, x, adjacent_matrix, weight_matrix):
        self.y = y
        self.x = x
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
            mu_phi = CAR('mu_phi', w=padded_weight_matrix, a=padded_adjacent_matrix, tau=tau_c, shape=N)

            # mu_total = pm.Deterministic('mu', )

            Yi = pm.Normal('Yi', mu=mu_phi + theta*x, sd=3,  observed=y)

            trace = pm.sample(5000, cores=2, tune=1000)

        return trace

if __name__ == '__main__':

    temp_data = pd.read_csv('./data/rain_and_flood_test.csv')

    temp_data = temp_data[~np.isnan(temp_data.GAGE_MAX)]
    temp_data = temp_data[~np.isnan(temp_data.PRCP)]

    from car_preprocess import ArealDataPreprocessor
    areal_data = ArealDataPreprocessor(temp_data, key='SITENUMBER')

    dataframe = areal_data.data
    adjacent_matrix = areal_data.adjacent_matrix
    weight_matrix = areal_data.weight_matrix
    padded_weight_matrix = areal_data.padded_weight_matrix
    padded_adjacent_matrix = areal_data.padded_adjacent_matrix
    x = dataframe['PRCP'].values
    y = dataframe['GAGE_MAX'].values
    N = len(x)


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


