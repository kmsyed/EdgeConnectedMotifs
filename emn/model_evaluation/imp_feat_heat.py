
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()


mpl.style.use('fivethirtyeight')

"""
plots heatmap for the feature importances of all 35 models: 35 plots
"""
mses_name = '../final_plots/ecoli_300_imp_feats.txt'
mses = pd.DataFrame.from_csv(mses_name, sep=',',index_col=0)
mses = mses.fillna(value=0)



uniform_data = mses.values

ax = sns.heatmap(uniform_data)

ax.show()

