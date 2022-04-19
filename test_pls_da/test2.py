# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:52:33 2022

@author: jihon
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target

k = [(i == 0) or (i == 1) for i in y]

X = X[k]
y = y[k]


from sklearn.cross_decomposition import PLSRegression
import seaborn as sns

pls = PLSRegression()
X_score, Y_score = pls.fit_transform(X, y)

plt.figure(dpi = 300)
sns.scatterplot(x = X_score[:, 0], y = X_score[:, 1], palette = 'Set2', hue = y)
plt.xlabel('PC1')
plt.ylabel('PC2')


import numpy as np

def calc_vip(model):
  t = model.x_scores_
  w = model.x_weights_
  q = model.y_loadings_
  p, h = w.shape
  vips = np.zeros((p,))
  s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
  total_s = np.sum(s)
  for i in range(p):
      weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
      vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
  return vips

vips = calc_vip(pls)
plt.figure(dpi = 300)
sns.heatmap(np.reshape(vips, (8,8)))
plt.show()

coefs = pls.coef_
plt.figure(dpi = 300)
sns.heatmap(np.reshape(coefs, (8,8)))
plt.show()


from sklearn.linear_model import ElasticNet
eln = ElasticNet()
eln.fit(X, y)

coefs = eln.coef_
plt.figure(dpi = 300)
sns.heatmap(np.reshape(coefs, (8,8)))
plt.show()



from sklearn.linear_model import LassoLars
lasso = LassoLars(alpha=0.02)
lasso.fit(X, y)

coefs = lasso.coef_
plt.figure(dpi = 300)
sns.heatmap(np.reshape(coefs, (8,8)))
plt.show()



