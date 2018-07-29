import pandas as pd
import numpy as np
from scipy.spatial import cKDTree as KDtree
from utilities_functions import coordinates_converter

class SpatialModelIDW:

    '''
    The basic idea is to borrow info from neighbor.
    df (pandas dataframe): needs to have the LATITUDE and LONGITUDE column
        and the column of variable of interest.

    response_var (str): Variable name
    spilt_ratio (float): Split ratio between train and test set
    K (int): number of neighbors to borrow information from
    '''

    def __init__(self, df, response_var = 'PRCP', spilt_ratio = 0.7, K = 5):

        self.response_var = response_var

        X = coordinates_converter(df).values
        y = df[self.response_var].values

        all_index = list(range(len(df)))
        train_size  = int(round(len(df)*spilt_ratio,0))

        train_index = np.random.choice(all_index, train_size)
        test_index = [idx for idx in all_index if idx not in train_index]

        self.X_train = X[train_index]; self.X_test = X[test_index]
        self.y_train = y[train_index]; self.y_test = y[test_index]

        self.train_loc_cache = df.loc[train_index, ['LATITUDE','LONGITUDE']]
        self.test_loc_cache = df.loc[test_index, ['LATITUDE','LONGITUDE']]

        self.K = K
        if self.K == 'auto':
            self.K = self._auto_choose_k(20)

    def fit(self):
        self.idw_model = KDtree(self.X_train)

    def _single_location_look_up(self, new_loc, new_real_value, p):

        dist, index = self.idw_model.query(new_loc, self.K)
        if 0 in list(dist):
            argmax = index[0]
            return self.y_train[argmax]

        weights = 1/(dist**p)
        standardized_weights = weights/np.sum(weights)
        prediction = sum(standardized_weights*self.y_train[index])
        return prediction

    def predict(self, new_df = None, distance_param = 2):

        if new_df:
            self.X_test = coordinates_converter(new_df).values
            self.y_test = new_df[self.response_var].values

        predictions = []
        for i, j in zip(self.X_test, self.y_test):
            prediction = self._single_location_look_up(i, j, p = distance_param)
            predictions.append(prediction)

        self.predictions = np.array(predictions)

        l1_loss = np.mean(np.abs(self.predictions - self.y_test))
        l2_loss = np.mean(np.square(self.predictions - self.y_test))
        self.summary = {'l1_loss': l1_loss, 'l2_loss': l2_loss}

        return self.predictions

    def _auto_choose_k(self, test_range):
        min_loss = 10000
        argmin = -1

        self.fit()

        for i in range(test_range):
            self.K = i + 2
            self.predict()
            loss = self.summary['l1_loss']
            if loss < min_loss:

                min_loss = loss
                argmin = self.K

        print(f'tested 2 to {test_range} neighbors, K = {argmin} gives the best MSE')
        return argmin

if __name__ == '__main__':

    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')

    idw_model = SpatialModelIDW(data, 'PRCP', K = 'auto')
    idw_model.fit()
    vars_ = idw_model.predict()

    import pickle
    with open('idw.pickle', 'wb') as handler:
        pickle.dump(idw_model, handler, protocol=pickle.HIGHEST_PROTOCOL)
    print(idw_model.summary)
