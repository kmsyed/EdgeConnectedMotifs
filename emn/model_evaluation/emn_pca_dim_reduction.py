"""
    Find the important features using Random Forest regression
    
"""


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
def random_forests_regerssion(dataset, target):
    """ make predictions using Random Forest regression model. """

    estimators = range(100, 300, 20)
    new_MSE = 0.0
    prev_MSE = 99999.99
    
    #selecting best estimator
    for eID, estimator in enumerate(estimators):
        model = RandomForestRegressor(n_estimators=estimator,
                                      bootstrap=True,
                                      oob_score=True,
                                      n_jobs=-1 ) 
        regr = model.fit(dataset, target)

        f_importances = regr.feature_importances_
        expected  = target
        
        predicted = regr.predict(dataset)
        COD = regr.score(dataset , target)
        new_MSE = mean_squared_error(predicted, expected)

        if new_MSE < prev_MSE :
            prev_MSE = best_MSE = new_MSE
            best_COD = COD

            best_features =  f_importances
            best_estimator = estimator

            imp_feat_ids = [(imp_id, imp_val) for imp_id, imp_val in enumerate (best_features)]
            imp_feat_ids.sort(key=lambda tup: tup[1], reverse=True)

            top_five_imp_feat = [item[0] for item in imp_feat_ids[:5]] 
            
    best_features = np.asarray(best_features)
    
    #reduced_feat = [imp_id+1 for imp_id, imp_val in enumerate (best_features) if imp_val < np.mean(best_features)]
    print ("RF: ",best_COD, best_estimator, imp_feat_ids[:5])

    return top_five_imp_feat


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

def find_final_ids():
    """ removes the duplicate record in featur set, """
    
    global data_final_ids, xdata, ydata
    data_final_ids = dict()
    
    xdata_np = np.array(xdata.values)
    xdata_hashable =  list(map(tuple, xdata_np))
    xdata_set = set(xdata_hashable)

    data_final_ids = dict()
    losses = ['ten', 'twen', 'threefive', 'fifty', 'sixty', 'sevenfive','nine']

    for lid, loss in enumerate(losses):
        net_dict =  collections.OrderedDict()
        for idx, rec in enumerate(xdata_hashable):
            net_dict.setdefault(rec, []).append(ydata.iloc[idx][loss])

        ydata_final_id = []
          
        for key in net_dict:
            #print(net_dict[key])
            if len(net_dict[key]) > 1:
                ydata_nid = sorted(net_dict[key])
                ydata_final_id.append(ydata_nid[0])
            else:
                ydata_final_id.append(net_dict[key][0])

        data_final_ids[loss] = ydata_final_id
   
        #print('---------------------------------------------------')
        

import numpy.ma as ma
def delete_feats(xdata_t, ydata_t):
    """deletes features with zero standerd deviation"""
    global xdata 
       
    matches = xdata_t.std(axis=0) > 0
    x_selected = xdata_t[matches[matches]]
    data_corr = np.corrcoef(x_selected)
    data_selec = ma.masked_where(data_corr < 0.95, data_corr)
    plt.pcolor(data_selec)
    plt.colorbar()
    plt.show()

def save_file(data_set, filen_name):
    xfile_name = '../VMNs/data/ecoli_{0}_features_no_dup.txt'.format(net_size)
    data_Set.to_csv(file_name, sep='\t', encoding='utf-8')


def main():
    """ main function processing begins here"""
    global xdata, ydata

    losses = ['ten','twen', 'threefive', 'fifty', 'sixty', 'sevenfive', 'nine']
    net_sizes = [100, 200, 300, 400, 500]

    for net_size in net_sizes:
        load_data(str(net_size))

        masked = xdata.duplicated(keep='first')
        print(len(xdata),  len(masked) - len(xdata[~masked]), len(xdata[~masked]) )

        find_final_ids()
            
        for lid, loss in enumerate(losses):
            xdata_ss , xdata_rs = data_scaling(xdata[~xdata.duplicated(keep='first')])
            #delete_feats(xdata_ss, ydata_final[loss])
            
            top_five = random_forests_regerssion(xdata_ss, data_final_ids[loss])
           
        #print('------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
