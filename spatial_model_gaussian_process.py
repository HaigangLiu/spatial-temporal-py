import pymc3 as pm
import numpy as np
from base_model import BaseModel

class GaussianProcessModel(BaseModel):
    '''
    Build a spatial model based on Pymc3 framework.
    Args:
        response (numpy array): 1d numpy array of response variable
        locations (numpy array): numpy n-d array for locations
            2d (lat and lon) and 3d (sphere) have been tested
        covariates (list of numpy arrays): a list of covariates, whose elements are 1d numpy arrays. Default value is None
    '''
    def __init__(self, response, locations, covariates=None):

        super().__init__(response, locations, covariates)
        print('Make sure latitude comes first in the locations column')

        self.number_of_covariates = len(self.covariates)
        self.dim_gp = locations.shape[1]

        if self.covariates:
            self.covariates.insert(0, np.ones([locations.shape[0], 1]))
        else:
            print('No covariates given.')

    @classmethod
    def from_panadas(cls, dataframe, response, locations, covariates=None):
        '''
        support specifying the model by column name
        dataframe (pandas): a dataframe
        response (string): name of the response variabel column
        locations (list): list of names of columns that specifies location
        covariates (list, NoneType by default): list of names of columns that specifies the covariates.
        '''
        if type(response)!= str:
            raise TypeError('the response variable has to be string of column name of repsonse varaible')
        if type(locations)!= list:
            raise TypeError('Locations has to be a list of column names that detemines the location')
        if covariates:
            if type(covariates)!= list:
                raise TypeError('Covariates has to be a list of column names that detemines the covariates or simply NoneType')
        try:
            reponse = dataframe[response].values
            locations = dataframe[locations].values
            if covariates:
                covariates_numpy = []
                for covariate in covariates:
                    covariates_numpy.append(dataframe[covariate].values)
            else:
                covariates_numpy = None
        except KeyError:
            print('make sure all keywords are in the columns of your dataframe')
            return None
        return cls(reponse, locations, covariates_numpy)

    def fit(self, sampling_size=5000, fast_sample=False):
        with pm.Model() as self.model:

            rho = pm.Exponential('rho', 1/5, shape=self.dim_gp)
            tau = pm.Exponential('tau', 1/3)

            cov_func = pm.gp.cov.Matern52(self.dim_gp, ls=rho)
            self.gp = pm.gp.Latent(cov_func=cov_func)
            f = self.gp.prior("f", X=self.locations)

            mean_func = f
            self.beta_list = []
            if self.covariates:
                for i in range(len(self.covariates)):
                    beta = pm.Normal('_'.join(['beta', str(i)]), mu=0, sd=50)
                    self.beta_list.append(beta)
                    mean_func = mean_func + beta*self.covariates[i]

            sigma = pm.HalfNormal('sigma', sd=20)
            y = pm.Normal('Y', mu=mean_func, sd=sigma, observed=self.response)

            if fast_sample:
                inference = pm.ADVI()
                approx = pm.fit(n=25000, method=inference) #until converge
                self.trace = approx.sample(draws=sampling_size)
            else:
                start = pm.find_MAP()
                self.trace = pm.sample(sampling_size, tune=10000, nchains=4)

    def predict(self, new_locations=None, new_covariates=None, sample_size=5000, use_median=False):

        if new_covariates is None:
            new_covariates = []

        if new_locations is None:
            print('no new locations are detected. Thus in-sample predictions are made')
            pred = self._predict_in_sample(sample_size, use_median)

        elif len(new_covariates) != self.number_of_covariates:
                print(f'{len(new_covariates)} != {self.number_of_covariates}')
                raise ValueError('the number of covariates must be equal to original data')
        else:
            pred = self._predict_out_of_sample(new_locations=new_locations, new_covariates=new_covariates, sample_size=sample_size, use_median=use_median)
        return pred

    def _predict_out_of_sample(self, new_locations, new_covariates, sample_size=5000, use_median=True):

        with self.model:
            gaussian_process_effect = self.gp.conditional("gp_effect", new_locations)

            if new_covariates: #if there is covariates
                for new_covariate, beta in zip(new_covariates, self.beta_list):
                    gaussian_process_effect = gaussian_process_effect + new_covariate*beta

            Y_new = pm.Deterministic('Y_new', gaussian_process_effect)
            simulated_values = pm.sample_ppc(self.trace, vars=[Y_new], samples= sample_size)['Y_new']

            if use_median:
                self.predicted_new = np.median(simulated_values, axis=0)
            else:
                self.predicted_new = np.mean(simulated_values, axis=0)
            return self.predicted_new

    def _predict_in_sample(self, sample_size, use_median):
        return super()._predict_in_sample(sample_size, use_median)

    def get_parameter_estimation(self, varnames, sig_level=0.95):
        return super().get_parameter_estimation(varnames, sig_level)

    def get_metrics(self):
        return super().get_metrics()

if __name__ == '__main__':
    from utility_functions import coordinates_converter
    from SampleDataLoader import load_rainfall_data

    data = load_rainfall_data('monthly')
    y = data['PRCP'].values
    locations = coordinates_converter(data[['LATITUDE', 'LONGITUDE']].values).values
    x1 = data[ 'SST'].values; x2 = data['ELEVATION'].values

    test_case_1 = False #with covariates, start from numpy; out of sample pred
    test_case_2 = False # no covariates, start from pandas; 2d vs 3d
    test_case_3 = True #with covariates, start from pandas; in sample pred
    test_case_4 = False # no covariate, from numpy

    if test_case_1:
        m1 = GaussianProcessModel(y, locations, [x1, x2])
        m1.fit(fast_sample=True)
        m1.get_metrics()
        m1.get_parameter_estimation(['beta_0', 'beta_1', 'beta_2'])
        pp = m1._predict_out_of_sample(locations[0:45], [x1[0:45], x2[0:45]])
        print(pp.shape)

    if test_case_2:
        m2 = GaussianProcessModel.from_panadas(data, 'PRCP', ['LATITUDE', 'LONGITUDE'])
        m2.fit(fast_sample=True)
        m2.get_parameter_estimation(['tau'])
        m2.get_metrics()

    if test_case_3:
        m3 = GaussianProcessModel.from_panadas(data, 'PRCP', ['LATITUDE', 'LONGITUDE'], ['ELEVATION'])
        m3.fit(fast_sample=True)
        m3.get_metrics()
        m3.get_parameter_estimation(['beta_0', 'beta_1'])
        m3.predict()

    if test_case_4:
        m4 = GaussianProcessModel(y, locations)
        m4.fit(fast_sample=True)
        m4.get_metrics()
        m4.get_parameter_estimation(['tau','rho'])
