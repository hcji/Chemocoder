# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 08:17:41 2022

@author: jihon
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

X = []
for i in range(5000):
    size = 200  # 信号长度为200
    xa = np.arange(0, size, 1)  # 信号x轴坐标
    noise = np.random.normal(0, 5, size)  # 随机噪声
    # 设置三个信号峰，位置和峰宽均随机
    rv1 = norm(loc = np.random.random(1) * 100, scale = np.random.random(1) * 20)  # 第一个峰
    rv2 = norm(loc = np.random.random(1) * 100, scale = np.random.random(1) * 20)  # 第二个峰
    rv3 = norm(loc = np.random.random(1) * 100, scale = np.random.random(1) * 20)  # 第三个峰
    x = (rv1.pdf(xa)+rv2.pdf(xa)+rv3.pdf(xa))*500 + noise 
    X.append(x)
X = np.array(X)
plt.plot(X[4])


import tensorflow as tf

dim = X.shape[1] # 输入维度
input_img = tf.keras.Input(shape=(dim,))  # 输入层

# 编码器
encoded = tf.keras.layers.Dense(32, activation='relu')(input_img)
encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(8, activation='relu')(encoded)

# 解码器
decoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(dim, activation='linear')(decoded)

# 模型编译
autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer = 'adam', loss = 'mse')
encoder = tf.keras.Model(input_img, encoded)

# 模型训练
autoencoder.fit(X, X,
                epochs=150,
                batch_size=128,
                shuffle=True)

# 信号重构
X_rebuild = autoencoder.predict(X)


plt.figure(figsize = (12,6), dpi = 300)
plt.subplot(231)
plt.plot(X[4])
plt.subplot(232)
plt.plot(X[7])
plt.subplot(233)
plt.plot(X[2])

plt.subplot(234)
plt.plot(X_rebuild[4])
plt.subplot(235)
plt.plot(X_rebuild[7])
plt.subplot(236)
plt.plot(X_rebuild[2])

