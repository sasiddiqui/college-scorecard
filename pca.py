# Sayeed Siddiqui
# Perform PCA on College Scorecard data for a simplified dataset

import pandas as pd
import numpy as np
import scipy.linalg as la 

Z = pd.read_csv('processed.csv').as_matrix()

cov = np.cov(Z, rowvar=False)
w, u = la.eigh(cov)

# Sort values in decreasing order
ind = np.argsort(w)[::-1]
u = u[:,ind]
w = w[ind]


