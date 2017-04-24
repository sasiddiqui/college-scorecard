# Sayeed Siddiqui
# Perform PCA on College Scorecard data for a simplified dataset

import pandas as pd
import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt

n = 100 

X = pd.read_csv('data/processed.csv')
Z = X.as_matrix()

Y = pd.read_csv('data/Y.csv', header=None)

cov = np.cov(Z, rowvar=False)
w, u = la.eigh(cov)

# Sort values in decreasing order
ind = np.argsort(w)[::-1][:n]
#u1 = u[:,ind]
#w1 = w[ind]

U = X.dot(u[:,ind[:35]])

# Evaluate PCA from 1 to 99 components
x = np.linspace(1,99)
y = [sum(w[ind[:int(i)]])/sum(w) for i in x]

fig, ax = plt.subplots()
ax.plot(x,y)
ax.set_xlabel('PCA components')
ax.set_ylabel('Variance explained')
#plt.show()

U.to_csv('data/pca.csv')
