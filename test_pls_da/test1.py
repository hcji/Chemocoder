# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:00:55 2022

@author: jihon
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X1, y1 = make_classification(n_samples = 200,          # 样本数
                             n_features = 60,         # 变量数 
                             n_informative = 2,        # 分类变量数
                             n_classes = 2,            # 有四个类别
                             n_clusters_per_class = 2, # 每个类别有1个簇
                             n_redundant = 10)         # 冗余变量数


X2, y2 = make_classification(n_samples = 200,          # 样本数
                             n_features = 60,         # 变量数 
                             n_informative = 2,        # 分类变量数
                             n_classes = 2,            # 有四个类别
                             n_clusters_per_class = 2, # 每个类别有1个簇
                             n_redundant = 20)         # 冗余变量数



X = np.hstack((X1, X2))






X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

pca = PCA()
embedding_pca = pca.fit_transform(X)

plt.figure(dpi = 300)
sns.scatterplot(x = embedding_pca[:, 0], y = embedding_pca[:, 1], palette = 'Set2', hue = y2)
plt.xlabel('PC1 ({} %)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)))
plt.ylabel('PC2 ({} %)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)))


plt.figure(dpi = 300)
sns.scatterplot(x = embedding_pca[:, 0], y = embedding_pca[:, 1], palette = 'Set2', hue = y1)
plt.xlabel('PC1 ({} %)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)))
plt.ylabel('PC2 ({} %)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)))


from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression()
X_score, Y_score = pls.fit_transform(X, y1)

plt.figure(dpi = 300)
sns.scatterplot(x = X_score[:, 0], y = X_score[:, 1], palette = 'Set2', hue = y1)
plt.xlabel('PC1')
plt.ylabel('PC2')




