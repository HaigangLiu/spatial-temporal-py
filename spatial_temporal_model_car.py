import pymc3 as pm
import numpy as np
import theano.tensor as tt

class CarModel:
    '''
    Fit a conditional autoregressive model (spatial-temporal model)
    This may NOT work with spatial data (not tested for that case);
    To fit spatial  model, use spatial_model module
    Args:
        response_var (np.array): 1-d array for response variable
        location_var: 1-d array to store location information
        covariates: nd array to store covariates; a column of constants will be added automatically
    '''
    def __init__(self, response_var, locations, covariates=None):

        self.response_var = response_var
        self.locations = locations
        self.N = self.response_var.shape[0] # N is number of locations
        self.number_of_days = response_var.shape[1]
        self.covariates = covariates

        if self.number_of_days <= 1:
            raise ValueError('the data only contains info of one day. Use spatial module instead')
        if self.covariates.shape[1]%self.number_of_days != 0:
            example_1 = f'if y is {self.N}x{self.number_of_days}, '
            example_2 = f'then the second dim of x has to be a muliple of {self.number_of_days}'
            raise ValueError(example_1 + example_2)
        else:
            self.dim = int(self.covariates.shape[1]/self.number_of_days)
        self.intercepts = np.ones((self.N, self.number_of_days))

        print('-'*20)
        print('BASIC INFO FROM INPUT')
        print(f'The sample size is {self.N}.')
        print(f'The time span is {self.number_of_days} days, and there are {self.dim} covariates in the model')
        print('Double check the input if any of these information does not seem right.')

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

    def fit(self, fast_sampling=True, sample_size=3000):

        if self.dim > 1:
            where_to_slice = [self.number_of_days*i for i in range(self.dim)]
            where_to_slice.pop(0)
            self.covariates_split = np.hsplit(self.covariates, where_to_slice)
        else:
            self.covariates_split = [self.covariates]

        with pm.Model() as self.model:
            beta_variables = []
            beta_names = []
            for i in range(self.dim+1):
                var_name = '_'.join(['beta', str(i)])
                beta_names.append(var_name)
                beta_variables.append(pm.Normal(var_name, mu=0.0, tau=1.0))

            # Priors for spatial random effects
            tau = pm.Gamma('tau', alpha=2., beta=2.)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            phi = pm.MvNormal('phi',
                              mu=0,
                              tau=tau*(self.D - alpha*self.weight_matrix),
                              shape=(self.number_of_days, self.N) #30 x 94 sample size by dim
                              )

            beta_0 = beta_variables.pop(0)
            mu_ = self.intercepts*beta_0 + phi.T
            for idx, beta_ in enumerate(beta_variables):
                mu_ += beta_*self.covariates_split[idx]

            # Mean model
            mu = pm.Deterministic('mu', mu_)
            theta_sd = pm.Gamma('theta_sd', alpha=1.0, beta=1.0)
            # Likelihood
            Yi = pm.Normal('Yi', mu=mu, tau=theta_sd, observed=self.response_var)

            if fast_sampling:
                inference = pm.ADVI()
                approx = pm.fit(n=25000, method=inference)
                self.trace = approx.sample(draws=sample_size)
            else:
                self.trace = pm.sample(sample_size, cores=2, tune=5000)
            for beta_name in beta_names:
                self.report_credible_interval(self.trace, beta_name)

    def report_credible_interval(self, trace, varname='beta'):
        mean = trace[varname].mean(axis=0)
        lower_bound = np.percentile(trace[varname], 2.5, axis=0)
        upper_bound = np.percentile(trace[varname], 97.5, axis=0)

        try: #in case param is nd array
            number_of_params = len(mean)
            for idx in range(number_of_params):
                print('-'*20)
                print(f'the mean of beta_{idx} is {mean[idx][0]}')
                print(f'the 95 percent credible interval for {varname}_{idx} is {lower_bound[idx][0], upper_bound[idx][0]}')
        except:
            #number_of_params = 1,  param is 1d array
            print('-'*20)
            print(f'the mean of {varname} is {mean}')
            print(f'the 95 percent credible interval for {varname} is {lower_bound, upper_bound}')

if __name__ == '__main__':

    import pandas as pd
    from data_preprocessing_tools import transpose_dataframe
    from utility_functions import get_in_between_dates

    start ='2014-01-01'; end = '2015-12-31'
    num_days = len(get_in_between_dates(start, end))

    checkout_df = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    df_sample_flat_wide = transpose_dataframe(checkout_df, start=start, end=end, time_varying_variables=['PRCP', 'DEV_GAGE_MAX', 'SPRING', 'SUMMER','FALL'])

    gage_level = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('DEV')]].values
    rainfall = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('PRCP')]].values
    elevations = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('ELEVATION')]].values
    elevations = np.tile(elevations, num_days)
    spring = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('SPRING')]].values
    summer = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('SUMMER')]].values
    fall = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('FALL')]].values

    locations = df_sample_flat_wide.BASIN.values
    covariate_m1 = rainfall
    covariate_m2 = np.hstack([rainfall, elevations])
    covariate_m3 = np.hstack([rainfall, elevations, spring, summer, fall])
    covariate_m4 = np.hstack([rainfall, elevations, rainfall*elevations ])

    if False:
        m1 = CarModel(covariates=covariate_m1,
                     locations=locations,
                    response_var=gage_level)
        m1.fit(fast_sampling=True, sample_size=5000)

    if False:
        m2 = CarModel(covariates=covariate_m2,
                     locations=locations,
                    response_var=gage_level)
        m2.fit(fast_sampling=True, sample_size=5000)

    if True:
        print('start fitting the third model')
        m3 = CarModel(covariates=covariate_m3,
                     locations=locations,
                    response_var=gage_level)
        m3.fit(fast_sampling=True, sample_size=5000)

    if True:
        print('start fitting the third model')
        m4 = CarModel(covariates=covariate_m4,
                     locations=locations,
                    response_var=gage_level)
        m4.fit(fast_sampling=True, sample_size=5000)
