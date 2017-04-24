# Sayeed Siddiqui
# Run multiple regression models, cross-validate hyperparameters
 
import pandas as pd
import numpy as np
import time
 
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
 
from sklearn.model_selection import GridSearchCV

X = pd.read_csv('data/pca.csv') 
Y = pd.read_csv('data/Y.csv', header=None).as_matrix().ravel()

preds = [(LinearRegression(),
            {}),
        (MLPRegressor(),
            {'activation': ['logistic','tanh','relu'],
             'hidden_layer_sizes': [[10],[7],[4]],
             'max_iter': [200]}),
        (SVR(),
            {'kernel': ['rbf','poly'],
             'degree': [1,2],
			 'C': [.5, 1]}),
        (Lasso(),
            {'max_iter': [100, 500, 1000],
             'alpha' : [.5, 1, 2]}),
        (KernelRidge(),
            {'kernel': ['rbf', 'poly'],
             'degree' : [2,3],
             'alpha' : [1, 2]})]

tpred, tparams = preds[4]

# Try all predictors, combinations of parameters, and different scores 
def evaluate(pred, params):
 
    for score in ['r2', 'neg_mean_squared_error']:
        start = time.time()
        CV = GridSearchCV(pred, params, cv=3, scoring=score)
        CV.fit(X, Y)
 
        print('%s was %0.2f with params: %s in %f s' % (score, CV.best_score_, str(CV.best_params_), time.time()-start))

    return CV
