from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import datasets
import matplotlib.pyplot as plt
mpl.style.use('fivethirtyeight')

"""
plots heatmap for the feature importances of all 35 models: 35 plots
"""
mses_name = '../results/reduced_features.txt'
mses = pd.DataFrame.from_csv(mses_name, sep=',',index_col=0)

print(mses.values)

feature_labels = mses.index

m, n = 3, 7

data = mses.values
for x in range(len(data)):
    for y in range(len(data[0])):
        plt.text( y, x, '%s' %data[x,y], horizontalalignment='center', verticalalignment='center')


ax = plt.imshow(mses, interpolation='nearest', cmap='Blues').get_axes()
cbar = plt.colorbar()
cbar.ax.set_ylabel('Reduced Feature Count', rotation=270)

cbar.ax.yaxis.set_label_coords(3.8, 0.5)
# reset limit for colorbar
#plt.clim(0, 16)

plt.xlabel('Loss %')
plt.ylabel('Network Sizes')
plt.xticks(rotation=70)
ax.set_xticks(np.linspace(0, n-1, n))
ax.set_xticklabels([10, 20, 35, 50, 60, 75, 90])
ax.set_yticks(np.linspace(0, m-1, m))
ax.set_yticklabels(feature_labels) # s, rotation=90)
ax.grid('off')



image_name = '../final_plots/' +'reduced_features_count_heatmap.pdf'

plt.savefig(image_name, format='pdf', bbox_inches='tight')
plt.close() #
