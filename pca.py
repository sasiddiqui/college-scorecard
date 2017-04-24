# Sayeed Siddiqui
# Perform PCA on College Scorecard data for a simplified dataset

from sklearn.decomposition import PCA

data = pandas.read_csv('cleaned.csv')
pca = PCA(n_components=2)
pca.fit(data)
