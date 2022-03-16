# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:24:29 2022

@author: jihon
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bar_chart_race as bcr

data = pd.read_excel('bar_chart_race/example_data.xlsx', index_col=0)

df = [np.nancumsum(data.iloc[:,i]) for i in range(data.shape[1])]
df = pd.DataFrame(df).T
df.columns = data.columns
df.index = data.index

df_values, df_ranks = bcr.prepare_wide_data(df, steps_per_period=4, 
                                            orientation='h', sort='desc')


bcr.bar_chart_race(df_values, filename='example.mp4', figsize=(7, 4),
                   n_bars=15, period_length=1000, dpi = 300)

