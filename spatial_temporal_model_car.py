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
    Fit a conditional autoregressive model (spatial-temporal model).
    To fit a pure spatial model, use spatial_model module
    Args:
        response (np.array): 1-d array for response variable
        location_var: 1-d array to store location information
        covariates: nd array to store covariates;
            intercepts will be generated automatically.
    '''
    def __init__(self, response, locations, covariates=None, autoreg=1):
        super().__init__(response, locations, covariates)

        assert response.ndim == 2, 'response should have 2 dimensions.'
        assert response.shape[1] > 1, 'only one days is detected. Use spatial module instead.'

        if covariates:
            assert type(covariates) == list, 'covariates has to be a list of numpy array'
            self.dim = len(covariates)
        else:
            covariates = []
            self.dim = 0

        self.N = response.shape[0]
        self.number_of_days = response.shape[1]

        if covariates:
            self.covariates = CarModel._covariate_handler(covariates, self.number_of_days)
        else:
            self.covariates = []

        self.shifted_response = []
        if autoreg: # redefine both x and y
            covariates_auto = [] #new X
            if self.covariates:
                for covariate in self.covariates:
                    covariate_remove_extra_days = covariate[:, autoreg:]
                    covariates_auto.append(covariate_remove_extra_days)
                self.covariates = covariates_auto; del covariates_auto

            for i in range(autoreg): #new Y_{t-1}
                right = self.number_of_days - autoreg + i
                make_shifted_y = self.response[:, i:right]
                self.shifted_response.append(make_shifted_y)

            self.response = self.response[:, autoreg:]
            self.number_of_days = self.number_of_days - autoreg
            print(f'autoregressive term is {autoreg}, and first {autoreg} day(s) will be used as covariates')

        print('-'*40)
        print('BASIC INFO FROM INPUT')
        print('-'*10)
        print(f'The sample size is {self.N}.')
        print(f'The time span is {self.number_of_days} days, and there are {self.dim} covariates in the model')
        print('Double check the input if any of these information does not seem right.')

        self.weight_matrix = None
        self.adjacent_matrix = None
        self._get_weight_matrices()

        self.residuals = None
        self.residual_by_region = None #average over stations in that region; time series data
        self.autoreg = autoreg

    @staticmethod
    def _covariate_handler(covariates, correct_dim):
        '''
        a helper function to make sure the dimension of input numpy arrays are correct
        three situations are considered:
         1. shape = (10, ): covert to (10, 1) then populate to (10,10)
            where 10 is the proper dimension for example.
        2. shape = (10 ,1), like first case, populate to (10, 10)
        3. shape = (10, 9) raise ValueError since data type not understood.
        '''
        covariates_ = [] #pure sanity check
        n = covariates[0].shape[0]

        for covariate in covariates:
            if n != covariate.shape[0]:
                raise ValueError('the length of covariates are not equal')
            if np.ndim(covariate) == 1:
                covariate = np.tile(covariate[:, None], correct_dim)
            elif np.ndim(covariate) == 2:
                if covariate.shape[1] == correct_dim:
                    pass
                elif covariate.shape[1] == 1:
                    covariate = np.tile(covariate, correct_dim)
                else:
                    raise ValueError(f'the proper dimension is {correct_dim}, get {covariate.shape[1]} instead')
            else:
                raise ValueError('dimension of covariates can only be either 1 or 2')
            covariates_.append(covariate)
        covariates_.append(np.ones((n, correct_dim)))
        return covariates_

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
            tau = pm.HalfNormal('tau', sd=100)
            alpha = pm.Uniform('alpha', lower=0, upper=1)
            self.phi = pm.MvNormal('phi',
                              mu=0,
                              tau=tau*(self.D - alpha*self.weight_matrix),
                              shape=(self.number_of_days, self.N) #the second dim has covar structure
                              )
            mu_ = self.phi.T

            if self.covariates: #add covars
                self.beta_variables = []; beta_names = []
                for idx, covariate in enumerate(self.covariates):
                    var_name = '_'.join(['beta', str(idx)])
                    beta_var = pm.Normal(var_name, mu=0.0, sd=100)
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

            theta_sd = pm.HalfNormal('theta_sd', sd=100)
            Y = pm.Normal('Y', mu=mu_, tau=theta_sd, observed=self.response)

    def get_residual_by_region(self):
        '''
        get residuals for each region.
        For one regions, there is usually multiple locations and multiple days,
        we average over different locations and thus got time series data for each big region.
        '''
        if self.predicted is None:
            _ = self.predict_in_sample(sample_size=1000, use_median=False)

        residuals = self.response - self.predicted #94*365
        resid_df = pd.DataFrame(residuals)
        resid_df['watershed'] = self.locations #94, 366
        self.residual_by_region = resid_df.groupby('watershed').mean() #8, 366
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
            fig = residual_by_region.T.plot(figsize=[figsize_baseline, figsize_baseline], alpha=0.7)
            fig  = fig.get_figure()

            if figname is None:
                from secrets import token_hex
                figname = 'tsplot_for_resid' + str(token_hex(2)) + '.png'
            # figname_ = '_'.join([figname,'tsplot_for_resid.png']) if figname else ''
            fig.savefig(figname)

        elif mode == 'histogram':
            fig = residual_by_region.T.hist(figsize=[figsize_baseline,figsize_baseline], alpha=0.7)
            # figname_ = '_'.join([figname,  '_histogram_for_resid.png']) if figname else 'histogram_for_resid.png'
            if figname is None:
                from secrets import token_hex
                figname = 'histogram_for_resid' + str(token_hex(2)) + '.png'
            plt.savefig(figname) #save the most recent

        else:
            raise ValueError("only allow two modes: 'time series' or 'histogram'." )

        full_dir  = os.path.join(os.getcwd(), figname)
        print(f'the image file has been saved to {full_dir}')
        f = Image.open(full_dir).show()

    def acf_by_region(self, figname=None, width=15, height=20, open_after_done=True):
        '''
        plot autoregressive functions
        '''
        if self.residual_by_region is None:
            _ = self.get_residual_by_region()

        if figname is None: #give a random name
            from secrets import token_hex
            random_token = token_hex(2)
            identifier = 'acf'
            name = identifier +  '-' + random_token + '.png'
        else:
            name = figname

        self._plot_rows(self.residual_by_region, acf, name, width, height, open_after_done)

    def pacf_by_region(self, figname=None, width=15, height=20, open_after_done=True):
        '''
        plot the partial autoregressive functions
        '''
        if self.residual_by_region is None:
            _ = self.get_residual_by_region()

        if figname is None:
            from secrets import token_hex
            random_token = token_hex(2)
            identifier = 'pacf'
            name = identifier + '-' + random_token + '.png'
        else:
            name = figname
        self._plot_rows(self.residual_by_region, pacf, name, width, height, open_after_done)

    def _plot_rows(self, dataframe, transformation, figname, width, height, open_after_done):
        '''
        internal utility function for acf and pacf
        plot rows in pandas dataframe individually
        allow transformation before plotting.
        allow differing length of data after transformation
        '''
        canvass_for_all = plt.figure(figsize=[width, height])
        rows = dataframe.index.tolist()
        num_of_figs = len(rows)

        for idx, arrays in enumerate(dataframe.values):
            graph_name = str(rows.pop(0))
            plt.subplot(num_of_figs, 1, idx+1)
            if transformation:
                tranformed_data = transformation(arrays)
            else:
                tranformed_data = arrays
            pd.Series(tranformed_data).plot(kind='bar', color='lightblue')
            plt.text(0.8,0.5, s=graph_name, horizontalalignment='left')

        full_dir = os.path.join(os.getcwd(), figname)
        canvass_for_all.savefig(full_dir)
        if open_after_done:
            f = Image.open(full_dir).show()

        print(f'the graph has been {figname} has been saved to {full_dir}')

    def predict_in_sample(self, sample_size=1000, use_median=False):
        self.predicted =  super()._predict_in_sample(sample_size=sample_size, use_median=use_median)
        return self.predicted

    def predict(self, new_x=None, steps=1, sample_size=1000, use_median=False):
        '''
        make predictions for future dates (out-of-sample prediction)
        Args:
            steps (int): how many days to forcast
            new_x(list): a list of numpy arrays. The dimension should match number of days
            sample_size (int): sample size of posterior sample
            use_median(boolean): if true, use median as point estimate otherwise mean will be used.
        '''
        if new_x:
            if not self.covariates:
                print('No covariates in original model; thus they should not show up in prediction.')
                print('hence input covariates igored')
                new_x = None
            else:
                shape = new_x[0].shape
                for numpy_array in new_x:
                    if shape != numpy_array.shape:
                        raise ValueError('the dimensions of covariates do not match!')
                else:
                    steps = shape[1] #1d sample size, 2d days
                new_x = CarModel._covariate_handler(new_x, steps)

        else:
            if self.covariates:
                raise ValueError('covariates not given. use predict_in_sample() if you want in-sample prediction.')
            else:
                print(f'now predicting future {steps} day(s). Change steps=10 if i.e., 10 terms are desired.')

        if self.autoreg > 0:
            _, y_temp = np.hsplit(self.response, [-self.autoreg])
            last_ys = np.hsplit(y_temp, range(self.autoreg))
            last_ys.pop(0)
        else:
            last_ys = None

        if steps >1 and new_x:
            x_split = []
            for covariate_ndarray in new_x:
                x_split_one_var = np.hsplit(covariate_ndarray, [i for i in range(steps)])
                x_split_one_var.pop(0)
                x_split.append(x_split_one_var)

            x_by_date = [] # make a list of cov by dates
            for day in range(steps):
                temp_var_one_day = []
                for covariate in x_split:
                    temp_var_one_day.append(covariate[day])
                x_by_date.append(temp_var_one_day)

            del temp_var_one_day
            del x_split

        def predict_one_step(current_x, last_y, name='Y_new'):
            '''
            the helper function to predit Yt+1
            will be called iteratively to predict multiple days
            '''
            with self.model:
                mean = self.phi.T[:, -1]
                if current_x: # a list of vars
                    for cov_, beta in zip(current_x, self.beta_variables):
                        mean = mean + cov_.ravel()*beta
                if last_ys is not None:
                    for last_y_, rho in zip(last_ys, self.rho_variables):
                        mean = mean + last_y_.ravel()*rho
                y_temp = pm.Deterministic(name, mean)
                svs = pm.sample_ppc(self.trace, vars=[y_temp], samples=sample_size)[name]
                if use_median:
                    y = np.median(svs, axis=0)
                else:
                    y = np.mean(svs, axis=0)
            return y

        if steps == 1:
            return predict_one_step(current_x=new_x, last_y=last_ys)

        elif steps > 1: #multipe steps
            y_history_rec = []
            idx = 1
            while steps:
                variable_name = '_'.join(['Y', str(idx)])
                if self.covariates:
                    current_x = x_by_date.pop()
                else:
                    current_x = None
                if self.autoreg > 0: #autoreg in model
                    y_most_recent = predict_one_step(current_x, last_ys, variable_name)
                    last_ys.append(y_most_recent)
                    last_ys.pop(0)
                else:
                    y_most_recent = predict_one_step(current_x, None, variable_name)
                y_history_rec.append(y_most_recent)
                steps -= 1; idx += 1
            return np.array(y_history_rec)
        else:
            raise ValueError('steps has to be a positive integer!')

if __name__ == '__main__':
    from data_preprocessing_tools import transpose_dataframe

    data_all = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    print(data_all.head())
    print(data_all.shape)
    retain = ['SITENUMBER','DATE','LATITUDE','LONGITUDE','ELEVATION', 'BASIN', 'PRCP', 'DEV_GAGE_MAX' ]
    data_all = data_all[retain]
    unstacked_df = data_all.set_index(['SITENUMBER', 'DATE'])
    count = data_all.groupby(['SITENUMBER']).count()
    print(count)
    # # data_all = data_all[['SITENUMBER', 'ELEVATION','DATE','DEV_GAGE_MAX','PRCP', 'BASIN', 'FALL', 'SPRING', 'SUMMER','FLOOD_SEASON']]
    # print(data_all.head())
    # print(data_all.columns.tolist())
    # train_fw = transpose_dataframe(data_all)
    # # test_fw = transpose_dataframe(data_all, start='2015-12-27', end='2015-12-31', fixed_variables=['ELEVATION'], time_varying_variables=['PRCP', 'DEV_GAGE_MAX', 'SPRING', 'SUMMER','FALL','FLOOD_SEASON'])


    # model_car_one_term = CarModel(covariates=[train_fw.PRCP.values],
    #               locations=train_fw.BASIN.values,
    #               response=train_fw.DEV_GAGE_MAX.values,
    #               autoreg=1)
    # model_car_one_term.fast_sample(iters=100000)


