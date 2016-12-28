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
import sys

#random Forest modeling
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error

#for reproducebility
np.random.seed(42)


def feature_reduction(xdata_train, xdata_test, imp_features):

    drops = []
    for fid, fval in imp_features:    
        if fval <= 0.03:
            drops.append(fid)

    for drop in drops:
        del xdata_train[drop]
        del xdata_test[drop]

    return xdata_train, xdata_test




def random_forests_regerssion(x_train, y_train, x_test, y_test, feat_names, loss, reduc_status):
    """ make predictions using Random Forest regression model. """

    global mses, feat_imp, feat_imp_bef

    estimators = range(100, 300, 10)
    new_MSE = 0.0
    prev_MSE = 99999.99
    best_MSE = 0.0
    mses_list = []

    new_oob_error = 0.0
    old_oob_error = 9999.99
    test_best_COD = 0.0
    test_prev_COD = -9999.0
    status = False

    

    #selecting best estimator
    for eID, estimator in enumerate(estimators):
        model = RandomForestRegressor(n_estimators=estimator,criterion="mse",
                                      bootstrap=True,
                                      oob_score=True,
                                      random_state=42,  
                                      max_features='sqrt',
                                      #max_depth=15,
                                      #min_samples_split=15,
                                      min_samples_leaf=5,
                                      n_jobs=-1 ) 
        
        regr = model.fit(x_train, y_train)

        f_importances = regr.feature_importances_

        predicted = regr.predict(x_train)

        new_MSE = mean_squared_error(predicted, y_train)
        test_COD = regr.score(x_test, y_test)


        #new_oob_error = 1 - regr.oob_score_
        #print(oob_error, new_MSE, estimator)
        if new_MSE < prev_MSE:
        #if  test_COD > test_best_COD :
        #if new_oob_error < old_oob_error :
            prev_MSE = best_MSE = new_MSE
            best_COD = regr.score(x_train, y_train)
            test_best_COD = test_prev_COD =  test_COD
            #old_oob_erro = new_oob_error
            #oob_score = regr.oob_score_
            best_features =  f_importances
            best_estimator = estimator
            status = True
        mses_list.append(new_MSE)


    mses[loss] = pd.Series(mses_list, index=estimators)

    del mses_list[:]

    if not status :
        print('In here getting best!!')
        best_features =  f_importances
        best_estimator = estimator
        best_COD = regr.score(x_train, y_train)
        test_best_COD =  test_COD
        best_MSE = new_MSE


    imp_feat_ids = [(feat_names[imp_id], imp_val) for imp_id, imp_val in enumerate (best_features)]
    imp_feat_ids.sort(key=lambda tup: tup[1], reverse=True)

    top_five_imp_feat = [item[0] for item in imp_feat_ids[:5]] 
     
          
    
    best_features = np.asarray(best_features)
    
    if reduc_status:
        #print(imp_feat_ids) 
        if len(feat_imp) == 0:
            feat_imp['feats'] = pd.Series(x_train.columns.values)
            feat_imp[loss] = pd.Series(best_features)
            
        else:
            df1 = pd.DataFrame()
            df1['feats'] = pd.Series(pd.Series(x_train.columns.values))
            df1[loss] = pd.Series(best_features)

            feat_imp = feat_imp.merge(df1, on='feats',how='outer')
    else:
        #print(imp_feat_ids) 
        if len(feat_imp_bef) == 0:
            feat_imp_bef['feats'] = pd.Series(x_train.columns.values)
            feat_imp_bef[loss] = pd.Series(best_features)
            
        else:
            df2 = pd.DataFrame()
            df2['feats'] = pd.Series(pd.Series(x_train.columns.values))
            df2[loss] = pd.Series(best_features)

            feat_imp_bef = feat_imp.merge(df2, on='feats',how='outer')

    #reduced_feat = [imp_id+1 for imp_id, imp_val in enumerate (best_features) if imp_val < np.mean(best_features)]
    print ("RF: ",best_COD, test_best_COD, best_estimator,  best_MSE, imp_feat_ids[:5])

    print(len([(idx, item)for idx, item in imp_feat_ids if item > 0.0]))
    return (imp_feat_ids, best_estimator, test_best_COD)


red_feat_counts = pd.DataFrame()
cods = pd.DataFrame()
mses = pd.DataFrame()

feat_imp = pd.DataFrame()
feat_imp_bef =pd.DataFrame()


def main():
    """ main function processing begins here"""
    global xdata, ydata, mses, cods, red_feat_counts

    losses = ['ten', 'twen','threefive', 'fifty', 'sixty', 'sevenfive', 'nine']
    net_sizes = [sys.argv[1]]# [300, 400, 500]

    for net_size in net_sizes:

        cod_bef = []
        cod_aft = []
        red_feats = []

        for lid, loss in enumerate(losses):
            print(net_size, loss)            
            xfile_name_train = '../data/train_test/ecoli_emn_{0}_{1}_train.txt'.format(net_size, loss)
            xfile_name_test = '../data/train_test/ecoli_emn_{0}_{1}_test.txt'.format(net_size, loss)

            xdata_train = pd.DataFrame.from_csv(xfile_name_train, sep=',',index_col=0)
            xdata_test = pd.DataFrame.from_csv(xfile_name_test, sep=',',index_col=0)

            ydata_train = pd.DataFrame()
            ydata_test = pd.DataFrame()

            ydata_train[loss] = xdata_train[loss]
            ydata_test[loss] = xdata_test[loss]

            del xdata_train[loss]
            del xdata_test[loss]

            ###-----------------------before feature reduction ---------------------------------

            feat_names = list(xdata_train.columns.values)
            print(net_size, loss, len(xdata_train)+len(xdata_test), len(xdata_train),len(xdata_test), len(feat_names))
            
            top_five, best_est, best_COD  = random_forests_regerssion(xdata_train, ydata_train[loss].values, xdata_test, ydata_test[loss].values, feat_names, '%s_%s_bef' %(net_size, loss), 0)

            
            cod_bef.append(best_COD)

            ###----------------------------after feature recduction-----------------------------
            xdata_train_red, xdata_test_red = feature_reduction(xdata_train, xdata_test, top_five)
            feat_names = list(xdata_train_red.columns.values)
            #print(feat_names)
            print(net_size, loss,len(xdata_train_red)+len(xdata_test_red),  len(xdata_train_red), len(xdata_test_red), len(feat_names))

            top_five, best_est, best_COD = random_forests_regerssion(xdata_train_red, ydata_train[loss].values, xdata_test_red, ydata_test[loss].values, feat_names, '%s_%s_aft' %(net_size, loss), 1)
            
            cod_aft.append(best_COD)
            red_feats.append(len(feat_names))

            print('--------------------------------------------------------------------------------------------')
        
        cods['%s_bef' %net_size] = pd.Series(cod_bef, index=losses)
        cods['%s_aft' %net_size] = pd.Series(cod_aft, index=losses)
        red_feat_counts[net_size] = pd.Series(red_feats, index=losses)
        
        print('#######################################################################################')

        xfile_name = '../results/ecoli_{0}_mses.txt'.format(net_size)
        mses.to_csv(xfile_name, sep=',', encoding='utf-8')


        feat_imp_name = '../final_plots/ecoli_{0}_imp_feats.txt'.format(net_size)
        feat_imp.to_csv(feat_imp_name, sep=',', encoding='utf-8')

        feat_imp_name_bef = '../final_plots/ecoli_{0}_imp_feats_bef.txt'.format(net_size)
        feat_imp_bef.to_csv(feat_imp_name_bef, sep=',', encoding='utf-8')

        #print(mses)
        print(cods)
        print(red_feat_counts)
        mses.drop(list(mses.columns.values), axis=1, inplace=True)
        feat_imp.drop(list(feat_imp.columns.values), axis=1, inplace=True)
        feat_imp_bef.drop(list(feat_imp.columns.values), axis=1, inplace=True)
    


if __name__ == '__main__':
    main()
