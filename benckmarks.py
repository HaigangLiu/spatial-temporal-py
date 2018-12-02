from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,  AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class ModelBuilder:
    '''
    a simple extension from sklean models
    used as benckmarks to compare with car models

    Base Model; not intended for end users to call
    '''
    def __init__(self, train_df, y, x_numeric, x_categorical):

        self.x_numeric = x_numeric
        self.x_categorical = x_categorical
        self.y = y
        self.train_df = train_df

        self.all_features = x_numeric.copy()
        self.all_features.extend(x_categorical)

    def build(self, model):
        steps_num = [('imputer', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler())]
        numeric_transformer = Pipeline(steps=steps_num)

        step_cat=[('imputer', SimpleImputer(strategy='constant')),
                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]
        categorical_transformer = Pipeline(steps=step_cat)

        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, self.x_numeric),
                          ('cat', categorical_transformer, self.x_categorical)])

        self.model = Pipeline(steps = [('prepare', preprocessor),
                                       ('model',  model)])

        response_var = self.train_df[self.y].values.ravel()
        self.model.fit(self.train_df[self.all_features], response_var)

    def predict(self, new_x):
        return self.model.predict(new_x)

    def report(self, true_y, predicted_y):
        '''
        report model fitting metrics based on new y
        '''
        reporter = []
        mse = mean_squared_error(true_y, predicted_y)
        mae = mean_absolute_error(true_y, predicted_y)
        r2score = r2_score(true_y, predicted_y)

        reporter.append(['mae', str(mae)])
        reporter.append(['mse', str(mse)])
        reporter.append(['r2score', str(r2score)])

        ModelBuilder.pretty_print('model fitting result',reporter)

    @staticmethod
    def pretty_print(header, table_to_print, table_len=40):
        '''
        header (string): header of table
        table_to_print (list of list): every list is a row.
            list element has to be string
        '''
        print('-'*table_len)
        print(header)
        col_width = max(len(word) for row in table_to_print for word in row) + 2  # padding
        for row in table_to_print:
            print('-'*table_len)
            print("".join(word.ljust(col_width) for word in row))
        print('-'*table_len)

class LinearModel(ModelBuilder):
    '''
    train_df: (pandas dataframe)
    y (string): name of column of response variable
    x_numeric (list): name(s) of column of covariates that are numeric
    x_categorical(list): name(s) of columns of covariates that are categorical
    '''
    def build(self,**kwargs):
    '''
    all kwargs in LinearRegression i.e., fit_intercept, are also allowed
    '''
        return super().build(LinearRegression(**kwargs))

class BoostingModel(ModelBuilder):
    '''
    train_df: (pandas dataframe)
    y (string): name of column of response variable
    x_numeric (list): name(s) of column of covariates that are numeric
    x_categorical(list): name(s) of columns of covariates that are categorical
    '''
    def build(self,**kwargs):
        '''
        all kwargs in AdaBoostRegressor i.e., random_state, are also allowed
        '''
        return super().build(AdaBoostRegressor(**kwargs))

class RandomForestModel(ModelBuilder):
    '''
    train_df: (pandas dataframe)
    y (string): name of column of response variable
    x_numeric (list): name(s) of column of covariates that are numeric
    x_categorical(list): name(s) of columns of covariates that are categorical
    '''
    def build(self,**kwargs):
    '''
    all kwargs in RandomForestRegressor i.e., n_estimators, are also allowed
    '''
        return super().build(RandomForestRegressor(**kwargs))

if __name__ == '__main__':
    import pandas as pd
    checkout_df = pd.read_csv('./data/check_out.csv', dtype={'SITENUMBER': str}, index_col=0)
    data_all = checkout_df[['DATE','DEV_GAGE_MAX','PRCP', 'BASIN', 'FALL', 'SPRING', 'SUMMER','FLOOD_SEASON']]
    train = data_all[ (data_all.DATE >= '2015-01-01') & (data_all.DATE <= '2015-12-26')]
    test = data_all[ (data_all.DATE >= '2015-12-27') & (data_all.DATE <= '2015-12-31')]

    if True:
        m1 = RandomForestModel(train, ['DEV_GAGE_MAX'], ['PRCP', 'FALL', 'SPRING',
            'SUMMER','FLOOD_SEASON'], ['BASIN'])
        m1.build(n_estimators=20)
        predicted = m1.predict(test[['PRCP', 'FALL', 'SPRING',
            'SUMMER','FLOOD_SEASON', 'BASIN']])
        m1.report(test['DEV_GAGE_MAX'].values, predicted)

    if True:
        m2 = LinearModel(train, ['DEV_GAGE_MAX'], ['PRCP', 'FALL', 'SPRING',
            'SUMMER','FLOOD_SEASON'], ['BASIN'])
        m2.build()
        predicted = m2.predict(test[['PRCP', 'FALL', 'SPRING',
            'SUMMER','FLOOD_SEASON', 'BASIN']])
        m2.report(test['DEV_GAGE_MAX'].values, predicted)

    if True:
        m3 = BoostingModel(train, ['DEV_GAGE_MAX'], ['PRCP', 'FALL', 'SPRING',
            'SUMMER','FLOOD_SEASON'], ['BASIN'])
        m3.build()
        predicted = m3.predict(test[['PRCP', 'FALL', 'SPRING',
            'SUMMER','FLOOD_SEASON', 'BASIN']])
        m3.report(test['DEV_GAGE_MAX'].values, predicted)

