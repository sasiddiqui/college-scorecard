# Sayeed Siddiqui
 
import pandas as pd
import numpy as np
 
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from RuleListClassifier import *
 
#from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
#from sklearn import metrics
 
clfs = [(tree.DecisionTreeClassifier(),
            {'criterion': ['gini', 'entropy'],
             'max_features': [1,2,3]}),
        (GaussianNB(),
            {}),
        (MLPClassifier(),
            {'activation': ['identity','logistic','tanh','relu'],
             'hidden_layer_sizes': [[10],[7],[4]],
             'max_iter': [500]}),
        (svm.SVC(),
            {'kernel': ['poly'],
             'degree': [1,2,3]})]
 
myc, myp = clfs[5]
 
# score is one of: accuracy, precision, recall, roc_auc, f1
def evaluate(clf, params):
 
    for score in ['accuracy', 'roc_auc', 'f1']:
        CV = GridSearchCV(clf, params, cv=10, scoring=score)
        CV.fit(X, C)
 
        print '%s was %0.2f with params: %s' % (score, CV.best_score_, str(CV.best_params_))
