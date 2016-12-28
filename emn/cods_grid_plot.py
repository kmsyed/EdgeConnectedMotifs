
import numpy as np
import  pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('fivethirtyeight')
# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.close('all')


# Four axes, returned as a 2-d array
f, axarr = plt.subplots(2, 2)

cods_name = '../results/300_cods.txt'
cods = pd.DataFrame.from_csv(cods_name, sep=',',index_col=0)

#cods = cods * 100

N = 7

net_sizes = [300] # , 400, 500]
for net_size in net_sizes:

	bef_name = '{0}_bef'.format(net_size)
	aft_name = '{0}_aft'.format(net_size)


	cods[bef_name] = cods[bef_name]#.round(2)
	cods[aft_name] = cods[aft_name]#.round(2)
	menMeans =[ item for item in  cods[bef_name].values]

	print(menMeans)


ind = np.arange(N)  # the x locations for the groups
width = 0.35    

rects1 = axarr[0, 0].bar(ind, menMeans, width, color='grey')

womenMeans = [ item for item in  cods[aft_name].values]

rects2 = axarr[0, 0].bar(ind + width, womenMeans, width, color='black')
print(ind + width+0.2)
print(ind + width)
axarr[0, 0].set_xticks(ind + width+0.05)
axarr[0, 0].set_xticklabels([10, 20, 35, 50, 60, 75, 90], rotation=0, fontsize=8)
axarr[0, 0].set_ylim(0.0, 1.0)
axarr[0, 0].set_yticks(np.arange(0.0,1.1,0.2 ))
axarr[0, 0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=10)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        axarr[0, 0].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom', rotation=90, fontsize=10)


autolabel(rects1)
autolabel(rects2)
axarr[0, 0].set_ylim(0, 1)
##################################################################
#400


cods_name = '../results/400_cods.txt'
cods = pd.DataFrame.from_csv(cods_name, sep=',',index_col=0)

net_sizes = [400] # , 400, 500]
for net_size in net_sizes:

	bef_name = '{0}_bef'.format(net_size)
	aft_name = '{0}_aft'.format(net_size)


	cods[bef_name] = cods[bef_name]#.round(2)
	cods[aft_name] = cods[aft_name]#.round(2)
	menMeans =[ item for item in  cods[bef_name].values]

	print(menMeans)


ind = np.arange(N)  # the x locations for the groups
width = 0.35    

rects1 = axarr[1, 0].bar(ind, menMeans, width, color='grey')

womenMeans = [ item for item in  cods[aft_name].values]

rects2 = axarr[1, 0].bar(ind + width, womenMeans, width, color='black')

axarr[1, 0].set_xticks(ind + width+0.05)
axarr[1, 0].set_xticklabels([10, 20, 35, 50, 60, 75, 90], rotation=0, fontsize=8)
#axarr[1, 0].set_xticklabels([10, 20, 35, 50, 60, 75, 90], rotation=0, fontsize=8)
axarr[1, 0].set_ylim(0.0, 1.0)
axarr[1, 0].set_yticks(np.arange(0.0,1.1,0.2 ))
axarr[1, 0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=10)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        axarr[1, 0].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom', rotation=90, fontsize=10)


autolabel(rects1)
autolabel(rects2)




####################################################


cods_name = '../results/500_cods.txt'
cods = pd.DataFrame.from_csv(cods_name, sep=',',index_col=0)

net_sizes = [500] # , 400, 500]
for net_size in net_sizes:

	bef_name = '{0}_bef'.format(net_size)
	aft_name = '{0}_aft'.format(net_size)


	cods[bef_name] = cods[bef_name]#.round(2)
	cods[aft_name] = cods[aft_name]#.round(2)
	menMeans =[ item for item in  cods[bef_name].values]

	print(menMeans)


ind = np.arange(N)  # the x locations for the groups
width = 0.35    

rects1 = axarr[1, 1].bar(ind, menMeans, width, color='grey')

womenMeans = [ item for item in  cods[aft_name].values]

rects2 = axarr[1, 1].bar(ind + width, womenMeans, width, color='black')


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        axarr[1, 1].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom', rotation=90, fontsize=10)


autolabel(rects1)
autolabel(rects2)
axarr[1, 1].set_xticks(ind + width+0.05)
axarr[1, 1].set_xticklabels([10, 20, 35, 50, 60, 75, 90], rotation=0, fontsize=8)

axarr[1, 1].set_ylim(0.0, 1.0)
axarr[1, 1].set_yticks(np.arange(0.0,1.1,0.2 ))
axarr[1, 1].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=10)

###############################################################
# add some text for labels, title and axes ticks
axarr[1, 0].set_ylim(0.0, 1.0)
axarr[1, 0].set_xlabel('Loss %',  fontsize=15)
axarr[1, 1].set_xlabel('Loss %',  fontsize=15)
axarr[0, 0].set_ylabel('Coefficient of Determination',  fontsize=12)
axarr[1, 0].set_ylabel('Coefficient of Determination',  fontsize=12)
#axarr[0, 0].set_title('Model Accuracies Before and After Feature Reduction for Networks Size {0}'.format(net_size), fontsize=12)
axarr[1, 0].set_xticks(ind + width+0.05)
axarr[1, 0].set_xticklabels([10, 20, 35, 50, 60, 75, 90], rotation=0, fontsize=8)

axarr[0, 1].legend((rects1[0], rects2[0]), ('All Features', 'Reduced Features'))


#axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)
override = {
    'fontsize'            : 'small',
    'verticalalignment'   : 'top',
    'horizontalalignment' : 'center'
    }


axarr[0, 0].set_title('300')
#axarr[0, 1].scatter(x, y)
axarr[1, 0].set_title('400')
#axarr[1, 0].plot(x, y ** 2)
axarr[1, 1].set_title('500')
axarr[0, 1].axis('off')
#axarr[1, 1].set_title('Axis [1,1]')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.savefig('../final_plots/all_cods.pdf',format='pdf', bbox_inches='tight')
plt.show()