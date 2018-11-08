import pymc3 as pm
import numpy as np
import theano.tensor as tt

class CarModel:
    '''
    Fit a conditional autoregressive model (spatial model)
    This will NOT work with spatial temporal data.
    To fit spatial temporal model, use spatial_temporal_model module
    Args:
        response_var (np.array): 1-d array for response variable
        location_var: 1-d array to store location information
        covariates: nd array to store covariates; a column of constants will be added automatically
    '''
    def __init__(self, response_var, locations, covariates=None):

        self.response_var = response_var
        self.locations = locations
        self.N = self.response_var.shape[0]
        self.number_of_days = response_var.shape[1]

        self.dim = int(covariates.shape[1]/self.number_of_days)
        self.covariates = np.hstack((covariates, np.ones((self.N, self.number_of_days))))

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

    # def _report_credible_interval(self, trace, varname='beta'):

    #     mean = trace[varname].mean(axis=0)
    #     lower_bound = np.percentile(trace[varname], 2.5, axis=0)
    #     upper_bound = np.percentile(trace[varname], 97.5, axis=0)

    #     try: #in case param is nd array
    #         number_of_params = len(mean)
    #         for idx in range(number_of_params):
    #             print('-'*20)
    #             print(f'the mean of beta_{idx} is {mean[idx][0]}')
    #             print(f'the 95 percent credible interval for {varname}_{idx} is {lower_bound[idx][0], upper_bound[idx][0]}')
    #     except:
    #         #number_of_params = 1,  param is 1d array
    #         print('-'*20)
    #         print(f'the mean of {varname} is {mean}')
    #         print(f'the 95 percent credible interval for {varname} is {lower_bound, upper_bound}')

    def fit(self, fast_sampling=True, sample_size=3000):

        with pm.Model() as self.model:
            beta = pm.Normal('beta', mu=0.0, tau=1.0, shape=60)
            # Priors for spatial random effects
            tau = pm.Gamma('tau', alpha=2., beta=2.)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            phi = pm.MvNormal('phi',
                              mu=0,
                              tau=tau*(self.D - alpha*self.weight_matrix),
                              shape=(self.number_of_days, self.N)
                              )
            # Mean model
            mu = pm.Deterministic('mu', tt.dot(self.covariates, beta) + phi.T)
            theta_sd  = pm.Gamma('theta_sd', alpha=1.0, beta=1.0)
            # Likelihood
            Yi = pm.Normal('Yi', mu=mu, tau=theta_sd, observed=self.response_var)

            if fast_sampling:
                inference = pm.ADVI()
                approx = pm.fit(n=50000, method=inference) #until converge
                self.trace = approx.sample(draws=sample_size)
            else:
                self.trace = pm.sample(sample_size, cores=2, tune=5000)

        # self._report_credible_interval(self.trace, 'beta')
        # self._report_credible_interval(self.trace, 'tau')

if __name__ == '__main__':
    import pandas as pd
    checkout_df = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    from data_preprocessing_tools import transpose_dataframe

    df_20_days = transpose_dataframe(checkout_df, start='2015-10-01',end='2015-10-30')
    yy = df_20_days[[i for i in df_20_days.columns if i.startswith('DEV')]].values
    xx = df_20_days[[i for i in df_20_days.columns if i.startswith('PRCP')]].values
    locations = df_20_days.BASIN.values

    #example 1 (one covariate)
    m1 = CarModel(covariates=xx,
                 locations=locations,
                response_var=yy)
    m1.fit(fast_sampling=False, sample_size=5000)
