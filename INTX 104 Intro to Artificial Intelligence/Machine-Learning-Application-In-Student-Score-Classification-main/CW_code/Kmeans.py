import numpy as np
import pandas as pd

from xgboost import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# 读取csv文件，删除ID列和空行
df = pd.read_csv('./CW_Data.csv', sep=',')
data = df.values
df.drop(labels="ID", axis=1, inplace=True)
df.drop(df[df['Programme'].isnull()].index, inplace=True)
df.drop_duplicates()#keep=False, subset=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], ignore_index=True , inplace=True)
X = np.array(df.drop(labels="Programme", axis=1))



# 2. 模型构建
k = 4
km = KMeans(n_clusters=k, init='k-means++', max_iter=30)
km.fit(X)
# 获取归集后的样本所属簇对应值
y_kmean = km.predict(X)

# pca = PCA(n_components = 2)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# numpy_name = np.array(df)
# X = pca.fit_transform(X)
# X = pca.fit_transform(numpy_name)

pca = PCA(n_components = 2)
pca.fit(X)
X2 = pca.fit_transform(X)
estimator = KMeans(n_clusters=k)
y_estimate = estimator.fit_predict(X2)

# tsne = TSNE(n_components=2)
# scaler = MinMaxScaler()
# X2 = scaler.fit_transform(X)
# X2 = tsne.fit_transform(X)

# nmf = NMF(n_components = 2)
# scaler = MinMaxScaler()
# X2 = scaler.fit_transform(X)
# X2 = nmf.fit_transform(X)
# stimator = KMeans(n_clusters=k)
# y_estimate = km.fit_predict(X2)
# 获取簇心
centroids = km.cluster_centers_

silhouette_avg = silhouette_score(X2, y_kmean, metric='euclidean')

# # 呈现未归集前的数据
plt.title("original data")
plt.scatter(X2[:, 0], X2[:, 1], s=50)
plt.yticks(())
plt.show()
# 画图

plt.suptitle("K-means(PCA modified)")
plt.title("n clusters= " + str(k) + "; silhouette score=" + str(silhouette_avg))
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5)
plt.scatter(X2[:, 0], X2[:, 1], c=y_kmean, s=50, cmap='viridis')
plt.show()

# 轮廓系数
score = []
for k in range(2, 20):
    cluster = KMeans(n_clusters=k, max_iter=500)
    cluster.fit(X)
    labels = cluster.predict(X)
    score_single = silhouette_score(X, labels, metric="euclidean")
    score.append(score_single)
x_index = np.linspace(3, 6, len(score))
plt.figure()
plt.title("silhouette score")
plt.plot(x_index, score)
plt.show()
