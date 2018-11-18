import pymc3 as pm
import numpy as np

class BaseModel:
    def __init__(self, response, locations, *covariates):
        '''
        Users can pass however many covariates, or no covariates at all.
        '''
        self.response = response
        self.locations = locations
        self.covariates = []

        for covariate in covariates:
            self.covariates.append(covariate)

        self.predicted = None
        self.trace = None
        self.model = None

    def get_metrics(self):

        if self.predicted is None:
            self.predict_in_sample()

        if self.predicted is None:
            self.predict_in_sample()

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

    def fit(self):
        '''
        need to generate three attributes when inherited:
        self.trace, self.model, and self.predicted
        '''
        pass

    def predict_in_sample(self, sample_size=5000, use_median=False):

        with self.model:
            simulated_values = pm.sample_ppc(self.trace)['Y']
        if use_median: #preferred, but might be pretty slow
            self.predicted = np.median(simulated_values, axis=0)
        else:
            self.predicted = np.mean(simulated_values, axis=0)
        return self.predicted

    def get_parameter_estimation(self, varnames, sig_level=0.05):
        if self.trace is None:
            self.fit()

        if isinstance(varnames, str):
            varnames = [varnames] # in case user passes in a string

        trace = self.trace
        output = []
        for varname in varnames:
            mean = trace[varname].mean(axis=0)
            lower_threshold = (100 - sig_level*100)/2
            upper_threshold = 100 - lower_threshold
            lower_bound = np.percentile(trace[varname], lower_threshold, axis=0)
            upper_bound = np.percentile(trace[varname], upper_threshold, axis=0)

            if upper_bound >= 0 and lower_bound >= 0:
                conclusion = '***'
            elif upper_bound <= 0 and lower_bound <= 0:
                conclusion = '***'
            else:
                conclusion = ' '

            mean = str(round(mean, 4))
            lower_bound = str(round(lower_bound, 4))
            upper_bound = str(round(upper_bound, 4))
            output.append([varname, mean, lower_bound, upper_bound, conclusion])
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

