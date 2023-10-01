import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('05 SVM과 Clustering/data/송파구_주택.txt', sep=',')

plt.scatter(x=data['x'], y=data['y'], s=1)
plt.show()

X = np.array(data.values)

#kmeans 클러스터링 진행
from sklearn.cluster import KMeans #kmeans
from scipy.spatial.distance import cdist


def display_cluster(X, y, centroid=None):
    plt.scatter(x=X[:,0], y=X[:,1], s=1, c=y)
    if centroid is not None :
        plt.scatter(x=centroid[:,0], y=centroid[:,1], s=30, color='black')
    plt.show()


for iter in range(1,10) :
    n_clusters= 8
    clt = KMeans(n_clusters=n_clusters, random_state=0, max_iter=iter)
    clt.fit(X)

    #중심점간 거리를 교차 계산
    centroid = clt.cluster_centers_.astype(np.int64())

    #각 픽셀별로 소속 클러스터로 맵핑한다.
    y = clt.predict(X)

    display_cluster(X, y, centroid)
    input('c')





from sklearn.cluster import DBSCAN


model = DBSCAN(eps=50, min_samples=10)
y = model.fit_predict(X)
display_cluster(X, y)