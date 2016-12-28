"""
    Find the important features using Random Forest regression
    
"""


from __future__ import division
#get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np 
from pandas import DataFrame
import matplotlib.pyplot as plt
from collections import Counter
import collections 

#random Forest modeling
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
import numpy as np


import time

import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


#data preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler

def data_scaling(X_train):
    """ Scaling  feature set"""

    # Standerd Scale
    standard_scaler = StandardScaler()
    Xtr_s = standard_scaler.fit_transform(X_train)

    # robust Scale 
    robust_scaler = RobustScaler(with_centering=False, copy=True, )
    Xtr_r = robust_scaler.fit_transform(X_train)

    return Xtr_s, Xtr_r

#global xdata(featureset) and ydata(target)
xdata = pd.DataFrame()
ydata = pd.DataFrame()

def load_data(net_size):
    """ loads the features(x) data and target(y_lable) data into dataframes"""
    global xdata, ydata
    
    #load x_data (feeture set)
    xdata = pd.DataFrame.from_csv('../data/ecoli_%s_features.txt' %net_size, sep=',',index_col=None)
    #print(xdata.head())

    #load target data (y label)
    ydata = pd.DataFrame.from_csv('../data/ecoli_%s_target.txt' %net_size,sep=',', index_col=None)
    #print(ydata.head())
   
from  collections import OrderedDict
data_final_ids = dict()

from sklearn.model_selection import train_test_split


def find_final_ids_pandas(xdata, ydata, loss):
    """ removes the duplicate record in featur set, """

    xdata_inter = pd.DataFrame()
    cols = list(xdata.columns.values)

    xdata_inter[cols] = xdata[cols]
    xdata_inter[loss] = ydata[loss]

    xdata_sorted = xdata_inter.sort_values([loss], ascending=False)
    
    mask = xdata_sorted.duplicated(cols[:-1], keep='first')
    xdata_final = xdata_sorted[~mask]
    ydata_final = pd.DataFrame()
    ydata_final[loss] = xdata_final[loss]
    del xdata_final[loss]

    return xdata_final, ydata_final
        
def save_file(data_set, filen_name):
    xfile_name = '../VMNs/data/ecoli_{0}_features_no_dup.txt'.format(net_size)
    data_Set.to_csv(file_name, sep='\t', encoding='utf-8')

import numpy.ma as ma

def remove_highly_corr_feats(xdata_t):#, ydata_t):
    """deletes features with zero standerd deviation"""

    xdata_t = xdata_t.drop(xdata_t.std()[xdata_t.std() == 0.0 ].index.values, axis=1)

    return xdata_t

def feature_reduction(xdata_train, xdata_test, imp_features):

    drops = []
    for fid, fval in imp_features:
        #print(fid, fval)
        #feat_name = 'feat%s' %(str(fid+1))
    
        if fval <= 0.019:
            #del xdata_old[fid]
            drops.append(fid)

    #xdata_train.drop(drops, axis=1)#, inplace=True)
    #xdata_test.drop(drops, axis=1)#, inplace=True)

    for drop in drops:
        del xdata_train[drop]
        del xdata_test[drop]

    return xdata_train, xdata_test

def emn_srv(x_train, y_train, x_test, y_test):

	clf = GridSearchCV(SVR(kernel='linear', gamma=0.1, epsilon=0.1), cv=4,
                   param_grid={"C": [1e0],"gamma": np.logspace(-2, 2, 5)})




	#clf = SVR(C=1.0, epsilon=0.2)
	#clf =  SVR(kernel='rbf', C=0.5, gamma=0.3, )
	clf.fit(x_train, y_train)
	#print(clf.best_params_)
	#print(clf.best_estimator_)
	print (clf.score(x_train, y_train))
	print (clf.score(x_test, y_test))


def main():
    """ main function processing begins here"""
    global xdata, ydata

    losses = ['ten','twen','threefive', 'fifty', 'sixty', 'sevenfive', 'nine']
    net_sizes = [100, 200, 300, 400, 500]


    for net_size in net_sizes:

        for lid, loss in enumerate(losses):
            load_data(str(net_size))


            ###------------------------after removing correlated featurs -----------------------
            xdata_uni, ydata_uni = find_final_ids_pandas(xdata, ydata, loss)
            xdata_corr= remove_highly_corr_feats(xdata_uni)

            xdata_corr[loss] = ydata_uni[loss]

            #xdata_train, xdata_test = train_test_split(xdata_corr, test_size = 0.10)

            xdata_train_uni = xdata_corr[~xdata_corr.duplicated([loss], keep='first')]
            xdata_rest = xdata_corr.drop(xdata_train_uni.index)

            test_size = min(0.99, round((len(xdata_corr)*0.2)/len(xdata_rest), 2))

            print(test_size)

            xdata_tr, xdata_test = train_test_split(xdata_rest, test_size = test_size)

            xdata_train = pd.concat([xdata_train_uni, xdata_tr])

            ydata_train = pd.DataFrame()
            ydata_test = pd.DataFrame()


            ydata_train[loss] = xdata_train[loss]
            ydata_test[loss] = xdata_test[loss]

            del xdata_train[loss]
            del xdata_test[loss]

            feat_names = list(xdata_train.columns.values)
            
            xdata_train_ss , xdata_train_rs = data_scaling(xdata_train)
            xdata_test_ss , xdata_test_rs = data_scaling(xdata_test)
            print(len(xdata_train_ss[1, :]))
            print(net_size, loss, len(xdata_corr), len(xdata_train_ss),len(ydata_train[loss].values), len(xdata_test), len(feat_names))

            emn_srv(xdata_train_ss, ydata_train[loss].values, xdata_test_ss, ydata_test[loss].values)

            #top_five, best_est  = random_forests_regerssion(xdata_train_ss, ydata_train[loss].values, xdata_test, ydata_test[loss].values)#, feat_names)
            #print(list(xdata.columns.values))

            """
            ###----------------------------after feature recduction-----------------------------
            xdata_train_red, xdata_test_red = feature_reduction(xdata_train, xdata_test, top_five)
            feat_names = list(xdata_train_red.columns.values)
            xdata_ss , xdata_rs = data_scaling(xdata_reduced)

            print(net_size, loss,len(xdata_corr),  len(xdata_train_red), len(xdata_test_red), len(feat_names))
            top_five = random_forests_regerssion(xdata_train_red, ydata_train[loss].values, xdata_test_red, ydata_test[loss].values, feat_names)
            """
            print('--------------------------------------------------------------------------------------------')
        
        print('#######################################################################################')

if __name__ == '__main__':
    main()
