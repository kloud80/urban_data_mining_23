import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



#임의의 분포를 만든다
X = np.random.normal(0,1,1000)

Y = X * 1 + np.random.normal(0,1,1000)

Y = np.random.normal(0,1,1000)

Y = X * -1 + np.random.normal(0, 1,1000)



#분포 출력
plt.scatter(X, Y)
plt.show()




#개별 분산 및 공분산을 계산한다.
공분산 = np.cov(X, Y)
print(공분산)



#공분산 행렬의 의미
균등좌표 = (np.arange(41) -20) * 0.1
균등좌표 = np.full([41,41], 균등좌표)
균등좌표 = np.concatenate([균등좌표[:,:, np.newaxis], 균등좌표.T[:,:, np.newaxis]], axis=2)


#균등좌표계 출력
plt.scatter(균등좌표[:,:,0], 균등좌표[:,:,1],s=0.5)
plt.show()



#공분산 행렬로 변환
변환좌표 = 균등좌표 @ 공분산



#변환된 좌표 출력
plt.scatter(변환좌표[:,:,0], 변환좌표[:,:,1],s=0.5)
plt.show()



#고유벡터
각도 = np.arange(360)
각도 = 각도 * np.pi / 180
각도_X = np.cos(각도)
각도_Y = np.sin(각도)
원래각도 = np.concatenate([각도_X[:,np.newaxis], 각도_Y[:,np.newaxis]], axis=1)

plt.scatter(원래각도[:,0], 원래각도[:,1],s=0.5)
plt.show()



#공분산행렬로 변환한다.
행렬변환 = 원래각도 @ 공분산

plt.scatter(행렬변환[:,0], 행렬변환[:,1],s=0.5)
plt.show()



def dist(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def getDegree(v):
    v1 = v[:2]
    v2 = v[2:]
    # 벡터 v1, v2의 크기 구하기
    distA = dist(v1)
    distB = dist(v2)

    # 내적 1 (x1x2 + y1y2)
    ip = v1[0] * v2[0] + v1[1] * v2[1]

    # 내적 2 (|v1|*|v2|*cos x)
    ip2 = distA * distB

    # cos x값 구하기
    cost = ip / ip2

    # x값(라디안) 구하기 (cos 역함수)
    x = math.acos(cost)

    # x값을 x도로 변환하기
    degX = math.degrees(x)
    return degX


기준벡터 = np.full([360,2], np.array([1,0]))
기준벡터 = np.concatenate([기준벡터, 행렬변환], axis=1)

변환각도 = np.apply_along_axis(getDegree, 1, 기준벡터) / 180 * np.pi

변환각도 = np.cos(변환각도)

각도차이 = 각도_X - 변환각도
각도차이 = 각도차이 * 각도차이

np.where(각도차이 < 0.0001)


plt.scatter(원래각도[:,0], 원래각도[:,1],s=0.5)
plt.scatter(원래각도[np.where(각도차이 < 0.0001),0], 원래각도[np.where(각도차이 < 0.0001),1],s=5, marker='X', color='#FF0000')
plt.show()


plt.scatter(행렬변환[:,0], 행렬변환[:,1],s=0.5)
plt.scatter(행렬변환[np.where(각도차이 < 0.0001),0], 행렬변환[np.where(각도차이 < 0.0001),1],s=5, marker='X', color='#FF0000')
plt.show()


plt.scatter(X, Y, alpha=0.2)
plt.scatter(행렬변환[:,0], 행렬변환[:,1],s=0.5)
plt.scatter(행렬변환[np.where(각도차이 < 0.00001),0], 행렬변환[np.where(각도차이 < 0.00001),1],s=20, marker='X', color='#FF0000')
plt.show()



"""===============================================================================
주성분 분석
==============================================================================="""

from sklearn.decomposition import PCA
data = np.concatenate([X[:,np.newaxis],Y[:,np.newaxis]], axis=1)

pca = PCA(n_components = 2) # feature 변수 개수가 2개
pca.fit(data)

pca.components_
pca.explained_variance_ratio_

predict = pca.transform(data)
pre2 = data @ pca.components_.T


plt.scatter(predict[:,0], predict[:,1])
plt.show()

