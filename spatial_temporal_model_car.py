import os
import theano
import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.tsa.stattools import acf, pacf

theano.config.compute_test_value = "ignore" #no default value error shall occur without this line
FAST_SAMPLE_ITERATION = 30 #advi setting

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

    def fit(self, fast_sampling=True, sample_size=3000):

        self.fitted_parameters = []
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
                approx = pm.fit(n=FAST_SAMPLE_ITERATION, method=inference)
                self.trace = approx.sample(draws=sample_size)
            else:
                self.trace = pm.sample(sample_size, cores=2, tune=5000)
            for beta_name in beta_names:
                self.fitted_parameters.append(self.get_interval(varname=beta_name))
            #report
            self.pretty_print(self.fitted_parameters)

    def get_metrics(self):
        '''
        generate a few metrics for model comparison
        '''
        print('-'*20)
        print('this is a summary of model metrics')
        if self.l1_loss is None:
            self.predict()
        print(f'The model mae is {self.l1_loss}')

        if self.l2_loss is None:
            self.predict()
        print(f'The model mse is {self.l2_loss}')

        self.result_waic = pm.waic(self.trace, self.model)
        self.result_leave_one_out = pm.loo(self.trace, self.model)
        print(f'the model waic is given by {self.result_waic}')
        print(f'the model leave-one-out accuracy is given by {self.result_leave_one_out}')
        print('all these  can be accessed as attributes by class-dot')
        print('-'*20)

    def get_residual_by_region(self):
        '''
        get residuals for each region.
        For one regions, there is usually multiple locations and multiple days,
        we average over different locations and thus got time series data for each big region.
        '''
        if self.residuals is None:
            self._predict_in_sample()

        resid_df = pd.DataFrame(self.residuals)
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
            figname = figname + '_tsplot_for_resid.png' if figname is None else 'tsplot_for_resid.png'
            fig.savefig(figname)

        elif mode == 'histogram':
            fig = residual_by_region.T.hist(figsize=[figsize_baseline,figsize_baseline], alpha=0.7)
            figname = figname + '_histogram_for_resid.png' if figname is None else 'histogram_for_resid.png'
            plt.savefig(figname) #save the most recent

        else:
            raise ValueError("only allow two modes: 'time series' or 'histogram'." )

        full_dir  = os.path.join(os.getcwd(), figname)
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
            full_dir = os.path.join([os.getcwd(), complete_name])
            canvass.savefig(full_dir)
            f = Image.open(full_dir).show()

        _plot(figname, type_='acf')
        _plot(figname, type_='pacf')

        # fig_pacf = plt.figure()
        # for idx, loc in enumerate(locs):
        #     plt.subplot(len(locs),1,idx + 1)
        #     pd.Series(pacf(residual_by_region.T[loc])).plot(kind='bar', color='lightblue')
        #     plt.text(0.8,0.5, s=loc, horizontalalignment='left')

        # name_pacf = figname + '_pacf.png' if figname else 'pacf.png'
        # dir_pacf = os.path.join(os.getcwd(), name_pacf)
        # fig_pacf.savefig(dir_pacf)
        # print(f'the image file has been saved to {dir_pacf}')

        # fig_acf = plt.figure(figsize=[fig_size, len(locs)*2.5])
        # for idx, loc in enumerate(locs):
        #     plt.subplot(len(locs),1,idx + 1)
        #     pd.Series(acf(residual_by_region.T[loc])).plot(kind='bar', color='lightblue')
        #     plt.text(0.8,0.5, s=loc, horizontalalignment='left')

        # name_acf = figname + '_acf.png' if figname else 'acf.png'
        # dir_acf = os.path.join(os.getcwd(), name_acf)
        # fig_acf.savefig(dir_acf)
        # print(f'the image file has been saved to {dir_acf}')

        # f1 = Image.open(dir_acf).show()
        # f2 = Image.open(dir_pacf).show()

    def predict(self, new_data=None, sample_size=1000, use_median=False):
        '''if there is no new data given, assume in-sample prediction. otherwise do it for new_x'''
        if new_data is None:
            print('no new data are given. In-sample predictions are made')
            self._predict_in_sample(sample_size=sample_size, use_median=use_median)
            return self.y_predicted_in_sample
        else:
            print('predictions are made based on given new data')
            self._predict_out_of_sample(new_x=new_data, sample_size=sample_size, use_median=use_median)
            return self.y_predicted_out_of_sample

    def _predict_in_sample(self, sample_size=1000, use_median=False):

        with self.model:
            simulated_values = pm.sample_ppc(self.trace)['Yi']
        if use_median: #might be pretty slow
            self.y_predicted_in_sample = np.median(simulated_values, axis=0)
        else:
            self.y_predicted_in_sample = np.mean(simulated_values, axis=0)
        self.residuals = self.response_var - self.y_predicted_in_sample

        #report mse and mae
        self.l1_loss = np.mean(np.abs(self.response_var - self.y_predicted_in_sample))
        self.l2_loss = np.mean(np.square(self.response_var - self.y_predicted_in_sample))
        print(f'The MAE of this model based on in-sample predictions is {self.l1_loss }')
        print(f'The MSE of this model based on in-sample predictions is {self.l2_loss}')

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
            simulated_values = pm.sample_ppc(self.trace)['Yi']
        self.y_predicted_out_of_sample = np.mean(simulated_values, axis=0)
        self.y_predicted_out_of_sample = self.y_predicted_out_of_sample[:, 0:days] #only first few column are relevant

    def get_interval(self, sig_level=0.95, varname='beta',  trace=None):

        trace = self.trace if trace is None else trace #default trace
        mean = trace[varname].mean(axis=0)

        self.lower_threshold = (100 - sig_level*100)/2
        self.upper_threshold = 100 - self.lower_threshold
        lower_bound = np.percentile(trace[varname], self.lower_threshold, axis=0)
        upper_bound = np.percentile(trace[varname], self.upper_threshold, axis=0)
        if upper_bound >= 0 and lower_bound >= 0:
            conclusion = '***'
        elif upper_bound <= 0 and lower_bound <= 0:
            conclusion = '***'
        else:
            conclusion = ' '
        # try: #in case param is nd array
        #     fitted_result = []
        #     number_of_params = len(mean)
        #     for idx in range(number_of_params):
        #         fitted_result.append([varname+'_'+str(idx), str(mean[idx][0]), str(lower_bound[idx][0]), str(upper_bound[idx][0])])
        #     return fitted_result

        # except:
        #number_of_params = 1,  param is 1d array
        mean = str(round(mean, 4))
        lower_bound = str(round(lower_bound, 4))
        upper_bound = str(round(upper_bound, 4))
        return [varname, mean, lower_bound, upper_bound, conclusion]

    def pretty_print(self, list_of_result):
        threshold_str_l = str(self.lower_threshold)
        threshold_str_u = str(self.upper_threshold)

        model_fitting_report = [['variable', 'point estimate', threshold_str_l,threshold_str_u, 'significant']]
        model_fitting_report.extend(list_of_result)

        print('-'*80)
        print('Model Fit Summary')
        col_width = max(len(word) for row in model_fitting_report for word in row) + 2  # padding
        for row in model_fitting_report:
            print('-'*80)
            print("".join(word.ljust(col_width) for word in row))
        print('-'*80)

if __name__ == '__main__':

    import pandas as pd
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
    covariate_m1 = rainfall
    covariate_m2 = np.hstack([rainfall, elevations])
    covariate_m3 = np.hstack([rainfall, elevations, spring, summer, fall])
    covariate_m4 = np.hstack([rainfall, elevations, rainfall*elevations, spring, summer, fall])
    covariate_m5 = np.hstack([rainfall, flood_season_indicator,flood_season_indicator*rainfall, elevations])
    covariates_m2_new = np.hstack([rainfall[:,3:5], elevations[:,3:5]])

    if False:
        m1 = CarModel(covariates=covariate_m1,
                     locations=locations,
                    response_var=gage_level)
        m1.fit(fast_sampling=True, sample_size=200)
        m1._predict_in_sample()

    if False:
        m2 = CarModel(covariates=covariate_m2,
                     locations=locations,
                    response_var=gage_level)
        m2.fit(fast_sampling=False, sample_size=1000)
        m2.predict(covariates_m2_new)
        m2.get_metrics()

    if False:
        print('start fitting the third model')
        m3 = CarModel(covariates=covariate_m3,
                     locations=locations,
                    response_var=gage_level)
        m3.fit(fast_sampling=False, sample_size=5000)
        m3.get_metrics()

    if True:
        print('start fitting the fourth model')
        m4 = CarModel(covariates=covariate_m5,
                     locations=locations,
                    response_var=gage_level)
        m4.fit(fast_sampling=True, sample_size=5000)
        # m4.predict()
        # m4.get_metrics()
        # m4.get_residual_by_region()
        m4.get_acf_and_pacf_by_region('garbage')
        m4.get_residual_plots_by_region('garbage')
