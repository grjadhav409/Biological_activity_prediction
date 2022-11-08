from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.svm import SVR
from utils import save_plot

class Model_Finder:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.file_object1 = open("Training_Logs/ModelTrainingLog1.txt", 'a+')
        self.logger_object = logger_object
        self.linearReg = LinearRegression()
        self.SVR = SVR()
        self.RandomForestReg = RandomForestRegressor()

    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):

        self.logger_object.log(self.file_object,
                               'Entered the RandomForestReg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_Random_forest_Tree = {
                "n_estimators": [10, 20, 30],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [2, 4, 8],
                "bootstrap": [True, False]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.RandomForestReg, self.param_grid_Random_forest_Tree, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.bootstrap = self.grid.best_params_['bootstrap']

            # creating a new model with the best parameters
            self.decisionTreeReg = RandomForestRegressor(n_estimators=self.n_estimators, max_features=self.max_features,
                                                         min_samples_split=self.min_samples_split,
                                                         bootstrap=self.bootstrap)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'RandomForestReg best params: ' + str(
                                       self.grid.best_params_) + '. Exited the RandomForestReg method of the Model_Finder class')
            return self.decisionTreeReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in RandomForestReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'RandomForestReg Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_linearReg(self, train_x, train_y):


        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_linearReg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_linearReg = {
                'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]

            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.linearReg, self.param_grid_linearReg, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.fit_intercept = self.grid.best_params_['fit_intercept']
            self.normalize = self.grid.best_params_['normalize']
            self.copy_X = self.grid.best_params_['copy_X']

            # creating a new model with the best parameters
            self.linReg = LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize,
                                           copy_X=self.copy_X)
            # training the mew model
            self.linReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'LinearRegression best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            return self.linReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_linearReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'LinearReg Parameter tuning  failed. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_SVR(self, train_x, train_y):

        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_SVR method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_SVR = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                              'gamma': ['scale', 'auto'],
                              'C': [0.001, 0.01, 0.1, 1, 10, 20]} # 50, 100
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.SVR, self.param_SVR, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.kernel = self.grid.best_params_['kernel']
            self.gamma = self.grid.best_params_['gamma']
            self.C = self.grid.best_params_['C']

            # creating a new model with the best parameters
            self.SVR = SVR(kernel=self.kernel, gamma=self.gamma,
                           C=self.C)
            # training the mew model
            self.SVR.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVR best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_SVR method of the Model_Finder class')
            return self.SVR
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_SVR method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'SVR Parameter tuning  failed. Exited the get_best_params_forSVR method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):

        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')

        try:
            # create best model for SVR
            self.SVR = self.get_best_params_for_SVR(train_x, train_y)
            self.prediction_SVR = self.SVR.predict(test_x)  # Predictions using the LinearReg Model
            self.SVR_error = r2_score(test_y, self.prediction_SVR)

            # create best model for XGBoost

            self.randomForestReg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)
            self.prediction_randomForestReg = self.randomForestReg.predict(
                test_x)  # Predictions using the randomForestReg Model
            self.prediction_randomForestReg_error = r2_score(test_y, self.prediction_randomForestReg)

            # save scores df
            scores = [self.SVR_error, self.prediction_randomForestReg_error]
            scores = sorted(scores, reverse=True)


            # Import writer class from csv module
            from csv import writer
            with open('results/Scores.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(scores)
                f_object.close()

            # comparing the two models, return best
            if (self.SVR_error < self.prediction_randomForestReg_error):
                return 'RandomForestRegressor', self.randomForestReg
            else:
                return 'SVR', self.SVR

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
