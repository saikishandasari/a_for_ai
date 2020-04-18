"""
=========================================================
Clustering Example
=========================================================
This example uses the `iris` dataset, in order to 
demonstrate a clustering technique. 
K-Means algorithm is used in the model.

Homogeneity, Completeness and V-measure scores are calculated.
"""
print(__doc__)

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

import matplotlib.pyplot as plt

import pandas as pd

# Get the data
iris = datasets.load_iris()

features = iris.data
labels = iris.target

# Find the best k - number of clusters
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(features)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
plt.show()

# Build the model
km = KMeans(n_clusters = 3)
predictions = km.fit_predict(features)

plt.scatter(features[:,0], features[:,1], c=predictions)
plt.show()

# Evaluate
print('Homogeneity Score: {}'.format(homogeneity_score(labels, predictions)))
print('Completeness Score: {}'.format(completeness_score(labels, predictions)))
print('V-Measure Score: {}'.format(v_measure_score(labels, predictions)))