import os
import numpy as np

import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def data_cleaning(data):
    data.drop(data[data['Programme'] == 0].index, inplace=True)  # remove major 0
    data.drop(data[data['Programme'].isnull()].index, inplace=True)  # remove empty rows
    data.drop_duplicates()  # keep=False, subset=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], ignore_index=True , inplace=True)  # remove noises
    return data


def read_CSV(path):
    data = pd.read_csv(path, sep=',', header=0)
    data.drop(labels="ID", axis=1, inplace=True)  # remove ID column
    print("read_file: " + path)
    return data


PATH = 'CW_Data.csv'
data = read_CSV(PATH)
data = data_cleaning(data)
data = data.values
scaler = MinMaxScaler()
dataset = scaler.fit_transform(data[:, 0:5])
labels_true = data[:, 5]
# print(labels_true)



cluster = KMeans(n_clusters=4, max_iter=500)
cluster.fit(dataset)
labels = cluster.predict(dataset)
score = silhouette_score(dataset, labels, metric='euclidean')
center = cluster.cluster_centers_
print(score)
tsne = TSNE(n_components=2)
dataset = tsne.fit_transform(dataset)
dataset = pd.DataFrame(dataset, columns=['X1', 'X2'])
labels = pd.DataFrame(labels, columns=["Label"])
Features = pd.concat([dataset, labels], axis=1)
# print(Features)




plt.figure()
plt.title("K-means++")
d = Features.loc[Features.Label == 0.0, :]
d = np.array(d)
for i in range(len(d)):
    plt.scatter(d[i][0], d[i][1], color='turquoise')
d = Features.loc[Features.Label == 1.0, :]
d = np.array(d)
for i in range(len(d)):
    plt.scatter(d[i][0], d[i][1], color='salmon')
d = Features.loc[Features.Label == 2.0, :]
d = np.array(d)
for i in range(len(d)):
    plt.scatter(d[i][0], d[i][1], color='gold')
d = Features.loc[Features.Label == 3.0, :]
d = np.array(d)
for i in range(len(d)):
    plt.scatter(d[i][0], d[i][1], color='violet')
d = Features.loc[Features.Label == 4.0, :]
d = np.array(d)
for i in range(len(d)):
    plt.scatter(d[i][0], d[i][1], color='cornflowerblue')
plt.show()

score = []
for k in range(2, 10):
    cluster = KMeans(n_clusters=k, max_iter=500)
    cluster.fit(dataset)
    labels = cluster.predict(dataset)
    score_single = silhouette_score(dataset, labels, metric='euclidean')
    score.append(score_single)
xindex = np.linspace(3, 6, len(score))
plt.figure()
plt.title("Silihouette Score")
plt.plot(xindex, score)
plt.show()