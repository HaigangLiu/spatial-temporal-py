import pymc3 as pm
import numpy as np
import theano.tensor as tt

import theano
theano.config.compute_test_value = "ignore" #no default value error shall occur without this line

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
        self.covariates = theano.shared(covariates)
        self.original_dim_cov = covariates.shape

        if self.number_of_days <= 1:
            raise ValueError('the data only contains info of one day. Use spatial module instead')
        if covariates.shape[1]%self.number_of_days != 0:
            example_1 = f'if y is {self.N}x{self.number_of_days}, '
            example_2 = f'then the second dim of x has to be a muliple of {self.number_of_days}'
            raise ValueError(example_1 + example_2)
        else:
            self.dim = int(covariates.shape[1]/self.number_of_days)
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
                self.report_credible_interval(varname=beta_name)

    def predict_in_sample(self, sample_size=1000, use_median=False):

        simulated_values = pm.sample_ppc(self.trace)['Yi']
        if use_median: #might be pretty slow
            self.y_predict_in_sample = np.median(simulated_values, axis=0)
        else:
            self.y_predict_in_sample = np.mean(simulated_values, axis=0)
        self.residuals = self.response_var - self.y_predict_in_sample

        #report mse and mae
        l1_loss = np.mean(np.abs(self.response_var - self.y_predict_in_sample))
        l2_loss = np.mean(np.square(self.response_var - self.y_predict_in_sample))
        print(f'The MAE of this model based on in-sample predictions is {l1_loss}')
        print(f'The MSE of this model based on in-sample predictions is {l2_loss}')


    def predict_new_data(self, new_x, days, sample_size=1000):

        if days < self.number_of_days:

            print('automatically padding the input to match dim of original covariates.')

            where_to_slice = [days*i for i in range(self.dim)]
            where_to_slice.pop(0)
            x_new_split = np.hsplit(new_x, where_to_slice)

            x_new_list_ = []
            for partial_data in x_new_split: #padding is required for all
                extra_days = self.number_of_days - days
                padded_values = np.random.standard_normal(self.N*d2).reshape(self.N, extra_days)
                new_x_i = np.hstack([partial_data, padded_values])
                x_new_list_.append(new_x_i)

            new_x = np.hstack(x_new_list_)
            new_x = np.hstack([new_x, self.intercepts])
            self.covariates.set_value(new_x)

        elif days == self.number_of_days:
            pass
        else:
            print('the days to predict is greater than the original dates. ')
            print('this use case is not supported.')
            return None

        simulated_values = pm.sample_ppc(self.trace)['Yi']
        self.y_predict_out_of_sample = np.mean(simulated_values, axis=0)


    def get_residual_by_location(self):
        pass



    def report_credible_interval(self, sig_level=0.95, varname='beta',  trace=None):

        trace = self.trace if trace is None else trace #default trace
        mean = trace[varname].mean(axis=0)

        lower_threshold = (100 - sig_level*100)/2
        upper_threshold = 100 - lower_threshold
        lower_bound = np.percentile(trace[varname], lower_threshold, axis=0)
        upper_bound = np.percentile(trace[varname], upper_threshold, axis=0)

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

    covariates_m2_new = np.hstack([rainfall[:,3], rainfall[:,3]])


    if False:
        m1 = CarModel(covariates=covariate_m1,
                     locations=locations,
                    response_var=gage_level)
        m1.fit(fast_sampling=True, sample_size=200)
        m1.predict_in_sample()

    if True:
        m2 = CarModel(covariates=covariate_m2,
                     locations=locations,
                    response_var=gage_level)
        m2.fit(fast_sampling=True, sample_size=5000)
        m2.predict_in_sample()
        m2.predict_new_data(covariates_m2_new)

    if False:
        print('start fitting the third model')
        m3 = CarModel(covariates=covariate_m3,
                     locations=locations,
                    response_var=gage_level)
        m3.fit(fast_sampling=False, sample_size=5000)

    if False:
        print('start fitting the third model')
        m4 = CarModel(covariates=covariate_m4,
                     locations=locations,
                    response_var=gage_level)
        m4.fit(fast_sampling=True, sample_size=5000)
