import numpy as np
from scipy.spatial import cKDTree as KDtree
from sklearn.model_selection import train_test_split

class InverseDistanceModel:
    '''
    Make predictions based on neigbouring values
    locations (numpy array): a n-d array input indicating the locations
    response (numpy array): a 1-d array with repsonse variable

    use from_pandas if users wish to build the model from pandas. See instructions below.
    '''
    def __init__(self, locations, response):
        self.response = response[:,None]
        self.locations = locations

    @classmethod
    def from_pandas(cls, dataframe, locations, response):
        '''
        build the models with a pandas dataframe
        dataframe (pandas dataframe with location infor)
        locations (list): name of the column(s) with location information
        response (string): name of the column with response variable
        '''
        try:
            r = dataframe[response].values
            l = dataframe[locations].values
            return cls(l, r)
        except KeyError:
            print('Double check the column names since they cannot be found in the dataframe')
            return None

    def fit(self, locations=None):
        if locations is None:
            self.model = KDtree(self.locations)
        else:
            self.model = KDtree(locations)

    def _predict_single_loc(self, new_loc, num_neighbors, p):
        # there is a value already in the location to predict, just use that value
        dist, index = self.model.query(new_loc, num_neighbors)

        try:
            if 0 in list(dist):
                argmax = index[0]
                return self.response[argmax]
        except TypeError:
            if dist == 0:
                argmax = index[0]
                return self.response[argmax]

        weights = 1/(dist**p)
        standardized_weights = weights/np.sum(weights)
        prediction = sum(standardized_weights*self.response[index])
        return prediction

    def predict(self, new_data, num_neighbors, p=2):
        predictions = []
        for row in new_data:
            prediction = self._predict_single_loc(row, num_neighbors=num_neighbors, p=p)
            predictions.append(prediction)
        self.predicted = np.array(predictions)
        return self.predicted.ravel()

    def select_best_k(self, test_size_ratio=0.4, p=2):
        X_train, X_test, y_train, y_test = train_test_split(self.locations, self.response, test_size=test_size_ratio)
        k_pool = [i + 1 for i in range(20)]

        records = []
        smallest_mse = 5000
        smallest_k = 0
        for k in k_pool:
            self.fit(X_train)
            predictions = self.predict(X_test, num_neighbors=k, p=p)
            mse = np.mean(np.abs(y_test - predictions))
            records.append(tuple([k, mse]))

            if mse < smallest_mse:
                smallest_mse = mse
                smallest_k = k

        print(f'the distance type is fixed throughout the test, which is p={p}.')
        print('And this can be changed by setting p=0.5 for instance')
        print(f'the smallest error occurs when number of num_neighbors={smallest_k}')
        print(f'the corresponding mse value is {smallest_mse}')
        print('-'*40)
        return records

if __name__ == '__main__':
    from SampleDataLoader import load_rainfall_data
    data = load_rainfall_data('monthly')

    idw_model = InverseDistanceModel( data[['LATITUDE', 'LONGITUDE']].values, data['PRCP'].values)
    s = idw_model.select_best_k(p=0.3)
    s = idw_model.select_best_k(p=0.5)
    s = idw_model.select_best_k(p=1)
    s = idw_model.select_best_k(p=1.5)
    s = idw_model.select_best_k(p=2)
