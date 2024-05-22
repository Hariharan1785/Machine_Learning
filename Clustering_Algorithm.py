import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import where, unique
from sklearn.datasets import make_classification

file = "C:\\Users\\USER\\OneDrive\\Top_Mentor\\Classes\\DataSets\\MachineLearning-master\\USArrests.csv"
df = pd.read_csv(file)
print(df)
X = df.iloc[:, 1:]
from sklearn.preprocessing import normalize

data_df = normalize(X)
data_df = pd.DataFrame(data_df)
print("Normalized the Data \n", data_df)

## Draw the dendogram
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(8, 5))
plt.title("USA")
dendo = sch.dendrogram(sch.linkage(data_df, method="ward"))
'''
ward method organizes datapoints into clusters (groups)
also known as Minimum Variance method

Normalize doesnt change the dataset but Standard scaling does
'''
plt.axhline(y=0.7)
plt.show()

## build our clustering model
# Agglomerative Clustering
print("Agglomerative_Clustering")
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, linkage="ward")
y_pred = cluster.fit_predict(data_df)
print("=========\ny_predicted for Agglomerative Clustering\n", y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()

# Birch Clustering
print("Birch_Clustering")
from sklearn.cluster import Birch

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = Birch(threshold=0.01, n_clusters=2)
y_pred = model.fit_predict(data_df)
print("=========\ny_predicted Birch\n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()


# Affinity Propagation

from sklearn.cluster import AffinityPropagation

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = AffinityPropagation(damping=0.9)
y_pred = model.fit_predict(data_df)
print("=========\ny_predicted Affinity Propagation\n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()


# DB Scan
from sklearn.cluster import DBSCAN

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = DBSCAN(eps=0.30,min_samples=9)
y_pred = model.fit_predict(data_df)
print("=========\ny_predicted DBScan \n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()


# Min Batch K-Means
from sklearn.cluster import MiniBatchKMeans

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = MiniBatchKMeans(n_clusters=2)
y_pred = model.fit_predict(data_df)
print("=========\ny_predicted MiniBatchK-Means \n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()

# MeanShift
from sklearn.cluster import MeanShift

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = MeanShift()
y_pred = model.fit_predict(data_df)
print("=========\ny_pred Mean-Shift \n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()

# Optics
from sklearn.cluster import OPTICS

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = OPTICS(eps=0.8,min_samples=10)
y_pred = model.fit_predict(data_df)
print("=========\n Y_Predicted for Optics \n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()

# Spectral Clustering
from sklearn.cluster import SpectralClustering

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = SpectralClustering(n_clusters=2)
y_pred = model.fit_predict(data_df)
print("=========\n Y_Predicted for Spectral Clustering \n", y_pred)
model = unique(y_pred)


# Gaussian Mixture
from sklearn.mixture import GaussianMixture

X = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                        random_state=2)

model = GaussianMixture(n_components=2)
y_pred = model.fit_predict(data_df)
print("=========\n Y_Predicted for GaussianMixture Clustering \n", y_pred)
model = unique(y_pred)
plt.scatter(data_df[0], data_df[1], c=cluster.labels_)
plt.show()
