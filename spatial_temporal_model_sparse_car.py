import scipy
import theano
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from spatial_temporal_model_car import CarModel
from pymc3.distributions import continuous
from pymc3.distributions import distribution

class SparseCar(distribution.Continuous):
    """
    Sparse Conditional Autoregressive (CAR) distribution
    Parameters
    ----------
    alpha : spatial smoothing term
    W : adjacency matrix
    tau : precision at each location
    """
    def __init__(self, alpha, W, tau, *args, **kwargs):
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.tau = tau = tt.as_tensor_variable(tau)
        D = W.sum(axis=0)
        n, m = W.shape
        self.n = n
        self.median = self.mode = self.mean = 0
        super(SparseCar, self).__init__(*args, **kwargs)

        # eigenvalues of D^−1/2 * W * D^−1/2
        Dinv_sqrt = np.diag(1 / np.sqrt(D))
        DWD = np.matmul(np.matmul(Dinv_sqrt, W), Dinv_sqrt)
        self.lam = scipy.linalg.eigvalsh(DWD)

        # sparse representation of W
        w_sparse = scipy.sparse.csr_matrix(W)
        self.W = theano.sparse.as_sparse_variable(w_sparse)
        self.D = tt.as_tensor_variable(D)

    def logp(self, x):
        logtau = self.n * tt.log(self.tau)
        logdet = tt.log(1 - self.alpha * self.lam).sum()

        Wx = theano.sparse.dot(self.W, x)
        tau_dot_x = self.D * x.T - self.alpha * Wx.ravel()
        logquad = self.tau * tt.dot(x.ravel(), tau_dot_x.ravel())
        return 0.5*(logtau + logdet - logquad)

class SparseCarModel(CarModel):

    def fit(self, sample_size=3000, sig=0.95):
        with pm.Model() as self.model:
            # Priors for spatial random effects
            tau = pm.HalfNormal('tau', sd=10)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            self.phi = SparseCar('phi',
                              alpha=alpha,
                              tau=tau,
                              W=self.weight_matrix,
                              shape=(self.N, 1) #30 x 94
                              # sample size by dim
                              )
            mu_ = theano.tensor.tile(self.phi, self.number_of_days)

            if self.covariates: #add covars
                self.beta_variables = []; beta_names = []
                for idx, covariate in enumerate(self.covariates):
                    var_name = '_'.join(['beta', str(idx)])
                    beta_var = pm.Normal(var_name, mu=0.0, sd=20)
                    beta_names.append(var_name)
                    self.beta_variables.append(beta_var)
                    mu_  = mu_ + beta_var*covariate

            if self.shifted_response: #add autoterms
                self.rho_variables = []; rho_names =[]
                autos = self.shifted_response.copy()

                for idx, autoterm in enumerate(reversed(autos)):
                    var_name = '_'.join(['rho', str(idx+1)])
                    rho_var = pm.Uniform(var_name, lower=0, upper=1)
                    rho_names.append(var_name)
                    self.rho_variables.append(rho_var)
                    mu_  = mu_ + rho_var*autoterm

            theta_sd = pm.HalfNormal('theta_sd', sd=10)
            Y = pm.Normal('Y', mu=mu_, tau=theta_sd, observed=self.response)
