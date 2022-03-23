# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:20:04 2022

@author: jihon
"""

import numpy as np
from sklearn.datasets import make_classification

X1, y = make_classification(n_samples = 200,     # 样本数
                           n_features = 2,       # 变量数 
                           n_informative = 2,    # 分类变量数
                           n_redundant = 0)      # 冗余变量数


X1 = X1 + 10
X2 = np.random.random((200, 1998)) + 10
X = np.hstack((X1, X2))

    
import seaborn as sns
import matplotlib.pyplot as plt

order = np.argsort(y)
plt.figure(dpi = 300)
sns.heatmap(X[order, :10], cmap='coolwarm')



from sklearn.preprocessing import StandardScaler
X_scl = StandardScaler(with_mean=True, with_std=False).fit_transform(X)


from sklearn.decomposition import PCA


pca = PCA()
embedding_pca = pca.fit_transform(X_scl)

plt.figure(dpi = 300)
sns.scatterplot(x = embedding_pca[:, 0], y = embedding_pca[:, 1], hue = y)
plt.show()