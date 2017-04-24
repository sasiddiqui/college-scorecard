# Sayeed Siddiqui
# Perform PCA on College Scorecard data for a simplified dataset

import pandas as pd
import numpy as np
import scipy.linalg as la 

n = 1

X = pd.read_csv('processed.csv')
Z = X.as_matrix()

Y = pd.read_csv('Y.csv', header=None)

cov = np.cov(Z, rowvar=False)
w, u = la.eigh(cov)

# Sort values in decreasing order
ind = np.argsort(w)[::-1][:n]
u = u[:,ind]
w = w[ind]

U = np.dot(Z, u)
