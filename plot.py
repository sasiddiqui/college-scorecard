# Sayeed Siddiqui
# 3-D plot of PCA components vs median debt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

dat = pd.read_csv('data/pca.csv') 
Z = pd.read_csv('data/Y.csv', header=None).as_matrix().ravel()

X = dat.iloc[:,1]
Y = dat.iloc[:,2]

s = ax.scatter(X, Y, Z, cmap=cm.coolwarm)

ax.set_xlabel('1st component')
ax.set_ylabel('2nd component')
ax.set_zlabel('Median debt')
plt.show()
