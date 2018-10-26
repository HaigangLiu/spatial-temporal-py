import pymc3 as pm
import numpy as np
import theano.tensor as tt
from pymc3.distributions import continuous
from pymc3.distributions import distribution

class CAR(distribution.Continuous):
    """
    Conditional Autoregressive (CAR) distribution

    Parameters
    ----------
    a : adjacency matrix
    w : weight matrix
    tau : precision at each location
    """
    def __init__(self, w, a, tau, *args, **kwargs):
        super(CAR, self).__init__(*args, **kwargs)
        self.a = a = tt.as_tensor_variable(a)
        self.w = w = tt.as_tensor_variable(w)
        self.tau = tau*tt.sum(w, axis=1)
        self.mode = 0.

    def logp(self, x):
        tau = self.tau
        w = self.w
        a = self.a
        mu_w = tt.sum(x*a, axis=1)/tt.sum(w, axis=1)
        return tt.sum(continuous.Normal.dist(mu=mu_w, tau=tau).logp(x))

class ArealModel:
    def __init__(self, response_var, covariates, locations):

        self.N = len(response_var)

        try:
            self.dim = covariates.shape[1]
            self.covariates = np.hstack((np.ones((self.N, 1)), covariates))
        except IndexError:
            self.dim = 1
            self.covariates = np.hstack((np.ones((self.N, 1)), covariates[:, None]))

        self.response_var = response_var[:,None]
        self.locations = locations[:,None]

        self.weight_matrix = None
        self.adjacent_matrix = None
        self._get_weight_matrices()

    def _get_weight_matrices(self):

        try:
            location_list = self.locations.tolist()
        except AttributeError:
            print('all inputs must be numpy.array type')
            return None

        neighbor_matrix = []
        weight_matrix = []
        adjacent_matrix = []

        for entry in location_list:
            w = [0 if entry!=comp else 1 for comp in location_list]
            neighbor_matrix.append(w)
        neighbor_matrix = np.array(neighbor_matrix)

        for idx, row in enumerate(neighbor_matrix):
            mask = np.argwhere(row == 1).ravel().tolist()
            mask.remove(idx) #delete the location itself.
            adjacent_matrix.append(mask)
            weight_matrix.append([1]*len(mask))

        wmat2 = np.zeros((self.N, self.N))
        amat2 = np.zeros((self.N, self.N), dtype='int32')

        for i, a in enumerate(adjacent_matrix):
            amat2[i, a] = 1
            wmat2[i, a] = weight_matrix[i]

        self.adjacent_matrix = amat2
        self.weight_matrix = wmat2
        self.D = np.diag(self.weight_matrix.sum(axis=1))

    def fit(self):
        with pm.Model() as self.model:
            beta = pm.Normal('beta', mu=0.0, tau=1.0, shape=(self.dim+1, 1))

            # Priors for spatial random effects
            tau = pm.Gamma('tau', alpha=2., beta=2.)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            phi = pm.MvNormal('phi',
                              mu=0,
                              tau=tau*(self.D - alpha*self.weight_matrix), shape=(1, self.N))
            # Mean model
            mu = pm.Deterministic('mu', tt.dot(self.covariates, beta) + phi.T)
            theta_sd  = pm.Gamma('theta_sd', alpha=1.0, beta=1.0)
            # Likelihood
            Yi = pm.Normal('Yi', mu=mu.ravel(), tau=theta_sd, observed=self.response_var)

            self.trace = pm.sample(1000, cores=2, tune=1000)


