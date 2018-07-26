import warnings
warnings.filterwarnings('ignore')
import pymc3 as pm
import numpy as np
import pandas as pd
from SSTcalculator import SSTcalculator
import matplotlib.pyplot as plt
from theano import shared

class GPModelSpatial:

    def __init__(self, year, month, dataform, response_var):


        df = dataform[(dataform.YEAR == year) & (dataform.MONTH == month) ]

        d1, d2, d3 = SSTcalculator._lat_lon_to_cartesian(df['LATITUDE'], df['LONGITUDE'])

        self.X = shared(np.array([d1, d2, d3]).T)
        self.y  = np.log(df[response_var].values)

    def build_gp_model(self, sampling_size = 5000, create_traceplot = True):

        with pm.Model() as model:

            rho = pm.Exponential('rho', 1, shape = 3)
            cov_func = pm.gp.cov.Matern52(3, ls = rho)

            gp = pm.gp.Marginal(cov_func = cov_func)
            sigma = pm.HalfCauchy("sigma", beta=3)
            y_ = gp.marginal_likelihood("y",
                                        X = self.X,
                                        y = self.y,
                                        noise = sigma)

            start = pm.find_MAP()
            self.trace = pm.sample(size = sampling_size)

        self.model = model
        if create_traceplot:
            pm.traceplot(trace, varnames = ['rho'])
            plt.show()

    def predict(self, new_data):

        self.X.set_value(new_data)
        with self.model:
            predicted_values = pm.sample_ppc(self.trace)
        return predicted_values


if __name__ == '__main__':

    data_form = pd.read_csv('/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/with_sst_5_years.csv')
    test_case = GPModelSpatial(2015, 10, data_form, 'PRCP')
    test_case.build_gp_model(create_traceplot = False)

    new_data = np.linspace(-1, 2, 900).reshape(300,3)
    vars = test_case.predict(new_data)

    print(vars)
