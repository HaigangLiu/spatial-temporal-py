import pymc3 as pm
import numpy as np

class BaseModel:
    def __init__(self, response, locations, covariates=None):

        for arg in [response, locations]:
            if isinstance(arg, np.ndarray):
                self.response = response
                self.locations = locations
            else:
                raise TypeError('Need input to be ndarray, or use classmethod from_panadas()')
        if covariates is None:
            self.covariates = [np.ones_like(response)]
        else:
            if isinstance(covariates, list):
                if BaseModel.covariate_is_legit(covariates):
                    covariates.insert(0, np.ones_like(response))
                self.covariates = covariates
            else:
                raise TypeError('covariates can be only either None or a list of numpy arrays')

        self.predicted = None
        self.trace = None
        self.model = None

    @staticmethod
    def covariate_is_legit(list_of_covars):
        '''
        make sure each covariate comes with identical dimension
        '''
        shape_info = list_of_covars[0].shape
        for covariate_arr in list_of_covars:
            shape_info_temp = covariate_arr.shape
            if shape_info == shape_info_temp:
                pass
            else:
                raise TypeError('All covariates should have the same dimension')
        return True

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
        if covariates and (type(covariates)!= list):
            raise TypeError('Covariates must be a list or NoneType')
        try:
            reponse = dataframe[response].values
            locations = dataframe[locations].values
            if covariates:
                cov_numpy_list = []
                for covariate in covariates:
                    cov_numpy_list.append(dataframe[covariate].values)
            else:
                cov_numpy_list = None
        except KeyError:
            print('make sure all keywords are in the columns of your dataframe')
            return None
        return cls(reponse, locations, cov_numpy_list)

    def sample(self, sample_size=5000, burn_in=1000, nchains=1):
        if self.model is None:
            self.fit()

        with self.model:
            self.trace = pm.sample(sample_size, tune=burn_in)

    def fast_sample(self, sample_size=5000, iters=10000):
        if self.model is None:
            self.fit()

        with self.model:
            inference = pm.ADVI()
            approx = pm.fit(n=iters, method=inference)#until converge
            self.trace = approx.sample(draws=sample_size)

    def fit(self):
        '''
        need to generate three attributes when inherited:
        self.trace, self.model, and self.predicted
        '''
        raise NotImplementedError('this is a base class. Should not be called in a normal use case.')

    def _predict_in_sample(self, sample_size, use_median):
        if self.model is None:
            self.fit()

        with self.model:
            simulated_values = pm.sample_ppc(self.trace)['Y']
        if use_median: #preferred, but might be pretty slow
            self.predicted = np.median(simulated_values, axis=0)
        else:
            self.predicted = np.mean(simulated_values, axis=0)
        return self.predicted

    def get_param_estimates(self, varnames, sig_level):

        lower_threshold = (100 - sig_level*100)/2
        upper_threshold = 100 - lower_threshold

        def draw_conclusion(lower_bound, upper_bound):
            if upper_bound > 0 and lower_bound > 0:
                return '***'
            elif upper_bound < 0 and lower_bound < 0:
                return '***'
            else:
                return ' '

        if self.trace is None:
            self.fit()

        if isinstance(varnames, str):
            varnames = [varnames] # in case user passes in a string

        trace = self.trace
        output = []

        for varname in varnames:
            temp = []
            lower_bounds = np.percentile(trace[varname], lower_threshold, axis=0)
            upper_bounds = np.percentile(trace[varname], upper_threshold, axis=0)
            means = trace[varname].mean(axis=0)

            try:
                packed_info = zip(means, lower_bounds, upper_bounds)
            except TypeError:
                packed_info = zip([means], [lower_bounds], [upper_bounds])

            for idx, entry in enumerate(packed_info):
                mean, lower_bound, upper_bound = entry
                conclusion = draw_conclusion(lower_bound, upper_bound)
                mean = str(round(mean, 4))
                lower_bound = str(round(lower_bound, 4))
                upper_bound = str(round(upper_bound, 4))
                temp.append([mean, lower_bound, upper_bound, conclusion])

            if len(temp) == 1:
                temp = temp[0]
                temp.insert(0, varname)
                output.append(temp)
            else:
                for idx, row in enumerate(temp):
                    varname_ = '_'.join([varname, str(idx)])
                    row.insert(0, varname_)
                    output.append(row)

        header = 'Parameter Estimates'
        BaseModel.pretty_print(content=output, header=header)
        output_dict = {} # build a dict for easier info retrval
        for entry in output:
            key =  entry.pop(0)
            values = entry
            output_dict[key] = values
        return output_dict

    def get_metrics(self, kind=['mse', 'mae', 'loo','aic'], sample_size=5000):
        if self.predicted is None:
            self._predict_in_sample(sample_size=sample_size, use_median=False)

        records = {}
        for kind_ in kind:
            if kind_.lower() == 'mse':
                records['mse'] = np.mean(np.square(self.response - self.predicted))
            elif kind_.lower() == 'mae':
                records['mae'] = np.mean(np.abs(self.response - self.predicted))
            elif kind_.lower() == 'waic':
                records['waic'] = pm.waic(self.trace, self.model).WAIC
            elif kind_.lower() == 'loo':
                records['loo'] = pm.loo(self.trace, self.model).LOO
            else:
                raise ValueError(f'{kind_} is not supported.')

        table_content = [['Metrics', 'Value']]
        for key, value in records.items():
            value = str(round(value, 4))
            table_content.append([key, value])

        header = 'Model Fitting Metrics Report'
        BaseModel.pretty_print(header, table, table_len=50)

    @staticmethod
    def pretty_print(header, content, table_len=80):
        '''
        header (string): header of table
        content (list of list): every list is a row.
            list element has to be string
        '''
        print('-'*table_len)
        print(header)
        col_width = max(len(word) for row in content for word in row) + 2  # padding
        for row in content:
            print('-'*table_len)
            print("".join(word.ljust(col_width) for word in row))
        print('-'*table_len)
