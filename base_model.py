import pymc3 as pm
import numpy as np

class BaseModel:

    def __init__(self, response, locations, covariates=None):
        '''
        Users can pass however many covariates, or no covariates at all.
        '''
        self.response = response
        self.locations = locations
        self.covariates = covariates

        if covariates is None: #no covariate information included
            self.covariates = []
        elif type(covariates) == list:
            pass
        else:
            raise TypeError('covariates must be a list of numpy arrays')
            return None

        self.predicted = None
        self.trace = None
        self.model = None

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

    def get_metrics(self):

        if self.predicted is None:
            self._predict_in_sample(sample_size=5000, use_median=False)

        waic = pm.waic(self.trace, self.model).WAIC
        loo = pm.loo(self.trace, self.model).LOO
        l1_loss = np.mean(np.abs(self.response - self.predicted))
        l2_loss = np.mean(np.square(self.response - self.predicted))

        table = [['Metrics', 'Value']]
        metrics = []
        metrics.append(['mse', str(round(l2_loss, 4))])
        metrics.append(['mae', str(round(l1_loss, 4))])
        metrics.append(['loo',str(round(loo, 4))])
        metrics.append(['aic', str(round(waic, 4))])
        table.extend(metrics)

        header = 'Model Fitting Metrics Report'
        BaseModel.pretty_print(header, table, table_len=50)

    def sample(self, sample_size=5000, burn_in=1000, nchains=4):
        if self.model is None:
            self.fit()

        with self.model:
            start = pm.find_MAP()
            self.trace = pm.sample(sampling_size, tune=burn_in,
                nchains=nchains)

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
        raise NotImplementedError('this is the fit method base class. Should not be called in common use case.')

    def _predict_in_sample(self, sample_size, use_median):

        with self.model:
            simulated_values = pm.sample_ppc(self.trace)['Y']
        if use_median: #preferred, but might be pretty slow
            self.predicted = np.median(simulated_values, axis=0)
        else:
            self.predicted = np.mean(simulated_values, axis=0)
        return self.predicted

    def get_parameter_estimation(self, varnames, sig_level):

        lower_threshold = (100 - sig_level*100)/2
        upper_threshold = 100 - lower_threshold

        def make_conclusion(lower_bound, upper_bound):
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
                conclusion = make_conclusion(lower_bound, upper_bound)
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
        BaseModel.pretty_print(table_to_print=output, header=header)
        output_dict = {} # build a dict for easier info retrval
        for entry in output:
            key =  entry.pop(0)
            values = entry
            output_dict[key] = values
        return output_dict

    @staticmethod
    def pretty_print(header, table_to_print, table_len=80):
        print('-'*table_len)
        print(header)
        col_width = max(len(word) for row in table_to_print for word in row) + 2  # padding
        for row in table_to_print:
            print('-'*table_len)
            print("".join(word.ljust(col_width) for word in row))
        print('-'*table_len)
