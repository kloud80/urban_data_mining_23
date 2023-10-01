# -*- coding: utf-8 -*-
"""
@author: Kloud
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.svm import SVC


mpl.rc('font', family='gulim') #한글 폰트 적용시

def SVM_margin(x, y, kernel='linear', c=1, g=1, d=0) :
    classifier = SVC(kernel = kernel, C=c, gamma = g, degree = d)
    classifier.fit(x, y) 
    classifier.decision_function(x)
    classifier.predict(x)
    
    
    
    fig, ax = plt.subplots(figsize = (12,12))
    plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
    plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
    plt.xlim(0,6)
    plt.ylim(0,6)
    plt.xticks(np.arange(0, 6, step=1))
    plt.yticks(np.arange(0, 6, step=1))
    
    
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, levels=[-1, 0, 1], alpha=0.5,
               colors=['#0000FF', '#000000', '#FF0000'], 
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=400,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()
    return classifier


x = np.array([[1, 2], [1, 5], [4, 1], [3, 5], [5, 5], [5, 2]])
y = np.array([1, 1, 1, -1, -1, -1])

fig = plt.figure(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))
plt.show()



classifier = SVM_margin(x, y)
classifier = SVM_margin(x, y, c=0.1)
classifier = SVM_margin(x, y, c=0.5)
classifier = SVM_margin(x, y, c=1)
classifier = SVM_margin(x, y, c=10)


classifier = SVM_margin(x, y, 'poly',d=1)
classifier = SVM_margin(x, y, 'poly',d=5)
classifier = SVM_margin(x, y, 'poly',d=10)
classifier = SVM_margin(x, y, 'poly',d=20)

classifier = SVM_margin(x, y, 'rbf', c=10, g=0.1, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.3, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.5, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.7, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.9, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=10.0, d=0)
