import os
import theano
import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.tsa.stattools import acf, pacf
from base_model import BaseModel

theano.config.compute_test_value = "ignore" #no default value error shall occur without this line
FAST_SAMPLE_ITERATION = 50 #advi setting

class CarModel(BaseModel):
    '''
    Fit a conditional autoregressive model (spatial-temporal model)
    This may NOT work with spatial data (not tested for that case);
    To fit spatial  model, use spatial_model module
    Args:
        response (np.array): 1-d array for response variable
        location_var: 1-d array to store location information
        covariates: nd array to store covariates; a column of constants will be added automatically
    '''
    def __init__(self, response, locations, covariates=None, autoreg=1):
        super().__init__(response, locations, covariates)

        try:
            self.N = response.shape[0]
            self.number_of_days = response.shape[1]
        except ValueError:
            print('Reshape your data either using array.reshape(-1, 1) if you data has single feature')
            return None

        covariates = [] if covariates is None else covariates

        if covariates:
            if type(covariates) != list:
                raise TypeError('covariates has to be a list of numpy array or NoneType by default')
            self.dim = len(covariates)
        else:
            self.dim = 0

        if self.number_of_days <= 1:
            raise ValueError('the data only contains one day. Use spatial module to avoid unexpected behavior')

        covariates_ = [] #pure sanity check
        for covariate in covariates:
            try:
                dim_cov = covariate.shape[1]
            except ValueError: #shape like (28, )
                covariate = np.tile(covariate[:, None], self.number_of_days)
                dim_cov = covariate.shape[1]

            if dim_cov == self.number_of_days:
                pass
            else:
                print(covariate.shape[1])
                print(self.number_of_days)
                raise ValueError('the number of cols must be equal num of days or equal to 1')
            covariates_.append(covariate)

        self.covariates = covariates_
        del covariates_

        self.autoregressive_terms = []

        if autoreg: # redefine both x and y
            covariates_auto = [] #new X
            for covariate in self.covariates:
                covariate_remove_extra_days = covariate[:, autoreg:]
                covariates_auto.append(covariate_remove_extra_days)
            self.covariates = covariates_auto; del covariates_auto

            for i in range(autoreg): #new Y_{t-1}
                right = self.number_of_days - autoreg + i
                make_autoreg_term = self.response[:, i:right]
                self.autoregressive_terms.append(make_autoreg_term)

            self.response = self.response[:, autoreg:]
            self.number_of_days = self.number_of_days - autoreg
            print(f'autoregressive term is {autoreg}, and first {autoreg} day(s) will be used as covariates')

        if self.covariates:
            self.intercepts = np.ones((self.N, self.number_of_days))
        else:
            self.intercepts = None

        print('-'*40)
        print('BASIC INFO FROM INPUT')
        print('-'*10)
        print(f'The sample size is {self.N}.')
        print(f'The time span is {self.number_of_days} days, and there are {self.dim} covariates in the model')
        print('Double check the input if any of these information does not seem right.')

        self.weight_matrix = None
        self.adjacent_matrix = None
        self._get_weight_matrices()

        self.l1_loss = None
        self.l2_loss = None

        self.residuals = None
        self.residual_by_region = None #average over stations in that region; time series data

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

    def fit(self, sample_size=3000, sig=0.95):

        with pm.Model() as self.model:

            # Priors for spatial random effects
            tau = pm.Gamma('tau', alpha=2., beta=2.)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            phi = pm.MvNormal('phi',
                              mu=0,
                              tau=tau*(self.D - alpha*self.weight_matrix),
                              shape=(self.number_of_days, self.N) #30 x 94 sample size by dim
                              )
            mu_ = phi.T

            if self.covariates: #add covars
                beta_variables = []; beta_names = []
                cov_with_intercept = self.covariates.copy()
                cov_with_intercept.append(self.intercepts)

                for idx, covariate in enumerate(cov_with_intercept):
                    var_name = '_'.join(['beta', str(idx)])
                    beta_var = pm.Normal(var_name, mu=0.0, tau=1.0)

                    beta_names.append(var_name)
                    beta_variables.append(beta_var)
                    mu_  = mu_ + beta_var*covariate

            if self.autoregressive_terms: #add autoterms
                rho_variables = []; rho_names =[]
                autos = self.autoregressive_terms.copy()

                for idx, autoterm in enumerate(reversed(autos)):
                    var_name = '_'.join(['rho', str(idx+1)])
                    rho_var = pm.Normal(var_name, mu=0.0, tau=1.0)

                    rho_names.append(var_name)
                    rho_variables.append(rho_var)
                    mu_  = mu_ + rho_var*autoterm
            # Mean model
            theta_sd = pm.Gamma('theta_sd', alpha=1.0, beta=1.0)
            # Likelihood
            Y = pm.Normal('Y', mu=mu_, tau=theta_sd, observed=self.response)

    def get_residual_by_region(self):
        '''
        get residuals for each region.
        For one regions, there is usually multiple locations and multiple days,
        we average over different locations and thus got time series data for each big region.
        '''
        if self.predicted is None:
            _ = self._predict_in_sample(sample_size=5000, use_median=False)

        residuals = self.response - self.predicted
        resid_df = pd.DataFrame(residuals)
        resid_df['watershed'] = self.locations
        self.residual_by_region = resid_df.groupby('watershed').mean()
        return self.residual_by_region

    def get_residual_plots_by_region(self, figname=None, mode='time series', figsize_baseline=10):
        '''
        provide either time series plot or histogram for residuals; split by different region.
        For each region, we find the mean of every time point
        if there are muliple observations for that regions and that day.
        '''
        if self.residual_by_region is None:
            _ = self.get_residual_by_region()

        residual_by_region = self.residual_by_region

        if mode == 'time series':
            fig = residual_by_region.T.plot(figsize=[figsize_baseline,figsize_baseline], alpha=0.7)
            fig  = fig.get_figure()
            figname_ = '_'.join([figname,'tsplot_for_resid.png']) if figname else 'tsplot_for_resid.png'
            fig.savefig(figname_)

        elif mode == 'histogram':
            fig = residual_by_region.T.hist(figsize=[figsize_baseline,figsize_baseline], alpha=0.7)
            figname_ = '_'.join([figname,  '_histogram_for_resid.png']) if figname else 'histogram_for_resid.png'
            plt.savefig(figname_) #save the most recent

        else:
            raise ValueError("only allow two modes: 'time series' or 'histogram'." )

        full_dir  = os.path.join(os.getcwd(), figname_)
        print(f'the image file has been saved to {full_dir}')
        f = Image.open(full_dir).show()

    def get_acf_and_pacf_by_region(self, figname=None, fig_size=15):

        if self.residual_by_region is None:
            _ = self.get_residual_by_region()

        residual_by_region = self.residual_by_region
        locs = list(set(self.locations))

        def _plot(figname, type_='acf'):
            canvass = plt.figure(figsize=[fig_size, len(locs)*2.5])
            for idx, loc in enumerate(locs):
                plt.subplot(len(locs),1,idx + 1)
                if type_ == 'acf':
                    pd.Series(acf(residual_by_region.T[loc])).plot(kind='bar', color='lightblue')
                else:
                    pd.Series(pacf(residual_by_region.T[loc])).plot(kind='bar', color='lightblue')
                plt.text(0.8,0.5, s=loc, horizontalalignment='left')
            figname_base = 'acf.png' if type_=='acf' else 'pacf.png'
            complete_name = '_'.join([figname, figname_base]) if  figname else figname_base
            full_dir = os.path.join(os.getcwd(), complete_name)
            canvass.savefig(full_dir)
            f = Image.open(full_dir).show()

        _plot(figname, type_='acf')
        _plot(figname, type_='pacf')

    def predict(self, new_data=None, sample_size=1000, use_median=False):
        '''if there is no new data given, assume in-sample prediction. otherwise do it for new_x'''
        if new_data is None:
            print('no new data are given. In-sample predictions are made')
            self._predict_in_sample(sample_size=sample_size, use_median=use_median)
            return self.predicted
        else:
            print('predictions are made based on given new data')
            self._predict_out_of_sample(new_x=new_data, sample_size=sample_size, use_median=use_median)
            return self.y_predicted_out_of_sample

    def _predict_out_of_sample(self, new_x, sample_size=1000,  use_median=False):
        try:
            days = int(new_x.shape[1]/self.dim)
        except IndexError: #just 1d this case
            new_x = new_x[:, None]

        if new_x.shape[1]%self.dim !=0:
            print(f'the new data has {new_x.shape[1]} columns')
            print(f'and there is {self.dim} variable')
            print(f'trying to imply the number of days involved but {new_x.shape[1]} over {self.dim} is not an int')
            raise ValueError('dimension mismatch: check the input dimension again')

        if days < self.number_of_days:
            print('automatically padding the input to match dim of original covariates.')
            print('this is due to a design choice of pymc3; in general this should not concern user')

            where_to_slice = [days*i for i in range(self.dim)]
            where_to_slice.pop(0)
            x_new_split = np.hsplit(new_x, where_to_slice)

            x_new_list_ = []
            for partial_data in x_new_split: #padding is required for all
                extra_days = self.number_of_days - days
                padded_values = np.random.standard_normal(self.N*extra_days).reshape(self.N, extra_days)
                new_x_i = np.hstack([partial_data, padded_values])
                x_new_list_.append(new_x_i)

            new_x = np.hstack(x_new_list_) # dont need intercept
            self.covariates.set_value(new_x)
        elif days == self.number_of_days:
            pass
        else:
            print('the days to predict is greater than the original dates. ')
            print('this use case is not supported.')
            return None

        with self.model:
            simulated_values = pm.sample_ppc(self.trace)['Y']
        self.y_predicted_out_of_sample = np.mean(simulated_values, axis=0)
        self.y_predicted_out_of_sample = self.y_predicted_out_of_sample[:, 0:days] #only first few column are relevant

if __name__ == '__main__':

    from data_preprocessing_tools import transpose_dataframe, mark_flood_season
    from utility_functions import get_in_between_dates

    start ='2015-01-01'; end = '2015-12-31'
    num_days = len(get_in_between_dates(start, end))

    checkout_df = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    checkout_df = mark_flood_season(checkout_df, start='2015-10-01', end='2015-12-31')

    df_sample_flat_wide = transpose_dataframe(checkout_df, start=start, end=end, time_varying_variables=['PRCP', 'DEV_GAGE_MAX', 'SPRING', 'SUMMER','FALL','FLOOD_SEASON'])

    gage_level = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('DEV')]].values
    rainfall = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('PRCP')]].values
    elevations = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('ELEVATION')]].values
    elevations = np.tile(elevations, num_days)

    flood_season_indicator = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('FLOOD_SEASON')]].values
    spring = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('SPRING')]].values
    summer = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('SUMMER')]].values
    fall = df_sample_flat_wide[[i for i in df_sample_flat_wide.columns if i.startswith('FALL')]].values

    locations = df_sample_flat_wide.BASIN.values
    covariate_m1 = [rainfall]
    covariate_m2 = [rainfall, elevations]
    covariate_m3 = [rainfall, elevations, spring, summer, fall]
    covariate_m4 = [rainfall, elevations, rainfall*elevations, spring, summer, fall]
    covariate_m5 = [rainfall, flood_season_indicator,flood_season_indicator*rainfall, elevations]

    if False:
        m1 = CarModel(covariates=[rainfall, elevations],
                     locations=locations,
                    response=gage_level)
        m1.fast_sample(iters=200)
        # m1._predict_in_sample()

    if True:
        m1 = CarModel(covariates=[rainfall, elevations],
                      locations=locations,
                      response=gage_level,
                      autoreg=2) #two terms of autoreg
        m1.fast_sample(iters=10)
        m1.predict()
        m1.get_metrics()
        m1.get_parameter_estimation(['beta_0', 'beta_1', 'rho_1','rho_2'], 0.95)
        m1.get_acf_and_pacf_by_region(figname='garbage')
        m1.get_residual_plots_by_region(figname='garbage')

    if False:
        m2 = CarModel(covariates=covariate_m2,
                     locations=locations,
                    response=gage_level)
        m2.fit(fast_sampling=False, sample_size=1000)
        m2.predict(covariates_m2_new)
        m2.get_metrics()

    if False:
        print('start fitting the third model')
        m3 = CarModel(covariates=covariate_m3,
                      locations=locations,
                      response=gage_level)
        m3.fit(fast_sampling=False, sample_size=5000)
        m3.get_metrics()

    if False:
        print('start fitting the fourth model')
        m4 = CarModel(covariates=covariate_m5,
                     locations=locations,
                    response=gage_level)
        m4.fit(fast_sampling=True, sample_size=5000)
        # m4.predict()
        m4.get_metrics()
        # m4.get_residual_by_region()
        m4.get_acf_and_pacf_by_region(figname='garbage')
        m4.get_residual_plots_by_region(figname='garbage')
