import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *

losses = ['ten', 'twen','threefive', 'fifty', 'sixty', 'sevenfive', 'nine']
net_sizes = [300, 400, 500]

for net_size in net_sizes:
    mses_name = '../results/ecoli_{0}_mses.txt'.format(net_size)
    mses = pd.DataFrame.from_csv(mses_name, sep=',',index_col=0)

    for loss in losses:

        # implement the example graphs/integral from pyx

        fig = plt.figure()
        ax = plt.subplot(111)
        bef_name = '{0}_{1}_bef'.format(net_size, loss)
        aft_name = '{0}_{1}_aft'.format(net_size, loss)

        plt.plot(mses.index, mses[bef_name], label='All Features')
        plt.plot(mses.index, mses[aft_name], label='Reduced Features')
        plt.xticks(rotation=70)
        ax.set_xticks(range(100,300, 10))
        ax.set_xticklabels(mses.index)
        #ax.grid('off')
        plt.xlabel('Estimators', fontsize=18)
        plt.ylabel('Mean Squared Error', fontsize=16)
        plt.legend(shadow=True, fancybox=True)


        #plt.show()
        plt.savefig('../final_plots/{0}_{1}_estimators.pdf'.format(net_size, loss),format='pdf', bbox_inches='tight')
    plt.close()        
