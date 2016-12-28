#!/usr/bin/env python
# a bar plot with errorbars

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *


import numpy as np
import matplotlib.pyplot as plt


import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import datasets
import matplotlib.pyplot as plt
mpl.style.use('fivethirtyeight')


cods_name = '../results/cods.txt'
cods = pd.DataFrame.from_csv(cods_name, sep=',',index_col=0)

cods = cods * 100

N = 7

net_sizes = [300, 400, 500]
for net_size in net_sizes:

	bef_name = '{0}_bef'.format(net_size)
	aft_name = '{0}_aft'.format(net_size)


	cods[bef_name] = cods[bef_name].round(2)
	cods[aft_name] = cods[aft_name].round(2)
	menMeans =[ item for item in  cods[bef_name].values]

	print(menMeans)


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, menMeans, width, color='grey')

	womenMeans = [ item for item in  cods[aft_name].values]

	rects2 = ax.bar(ind + width, womenMeans, width, color='black')

	# add some text for labels, title and axes ticks
	ax.set_xlabel('Loss %',  fontsize=18)
	ax.set_ylabel('Coefficient of Determination',  fontsize=16)
	ax.set_title('Model Accuracies Before and After Feature Reduction for Networks Size {0}'.format(net_size), fontsize=12)
	ax.set_xticks(ind + width)
	ax.set_xticklabels([10, 20, 35, 50, 60, 75, 90])

	ax.legend((rects1[0], rects2[0]), ('All Features', 'Reduced Features'))

	plt.ylim([0, 100])

	def autolabel(rects):
	    # attach some text labels
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                '%.1f' % height,
	                ha='center', va='bottom', rotation=70, fontsize=10)

	autolabel(rects1)
	autolabel(rects2)

	plt.savefig('../final_plots/{0}_cods.pdf'.format(net_size),format='pdf', bbox_inches='tight')

	#plt.show()
	plt.close()