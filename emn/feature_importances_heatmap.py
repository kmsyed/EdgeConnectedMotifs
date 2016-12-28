
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

net_size =sys.argv[1]

mses_name = '../results/ecoli_{0}_imp_feats.txt'.format(net_size)
mses = pd.DataFrame.from_csv(mses_name, sep=',',index_col=0)
mses = mses.fillna(value=0)

xlabels = [item.replace('feat','') for item in mses['feats'].values]
ylabels = ['10', '20','35', '50', '60', '75', '90']
feature_labels = xlabels

mses.drop(['feats'], axis=1, inplace=True)

data = np.array(mses.values)

data = data.transpose()

print(data)



m, n =  len(mses.columns.values), len(xlabels)


ax = plt.imshow(data, interpolation='nearest',origin='lower',  cmap='Blues', aspect='auto').get_axes()
cbar = plt.colorbar()
cbar.ax.set_ylabel('Feature Importance', rotation=270)

cbar.ax.yaxis.set_label_coords(3.8, 0.5)
# reset limit for colorbar
#plt.clim(0, 16)

plt.title('Reduced Features Importance for Size {0}'.format(net_size))
plt.ylabel('Loss %')
plt.xlabel('Feature IDs')
plt.xticks(rotation=70)
ax.set_xticks(np.linspace(0, n-1, n))
ax.set_xticklabels(xlabels, rotation=75)
ax.set_yticks(np.linspace(0, m-1, m))
ax.set_yticklabels(ylabels) # s, rotation=90)
ax.grid('off')



image_name = '../final_plots/' +'{0}_features_importance_heatmap.pdf'.format(net_size)

plt.savefig(image_name, format='pdf', bbox_inches='tight')
plt.close() #
