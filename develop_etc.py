# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 10:19:23 2017

@author: mtinti-x
"""


import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_selection import RFECV
from bayes_opt import BayesianOptimization
from sklearn import ensemble
from sklearn.cross_validation import KFold
import os
from utility import get_predictor_name
from string import strip

#avaiable scores

# 'adjusted_rand_score', 'average_precision', 
# 'f1', 'f1_macro', 'f1_micro',  'f1_weighted', 'neg_log_loss', 
#'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 
#'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
# 'r2', 'recall', 'recall_macro', 'recall_micro',  'recall_weighted', 'roc_auc'


def make_RFECV(scoring = '', rfc='', X='', y='', index=0, path_save=''):
    
    rfc_name = get_predictor_name(rfc)

    rfecv = RFECV(estimator=rfc,
              step=100,
              cv=KFold(y.shape[0],
                       n_folds=5,
                       shuffle=True,
                       random_state=101),
              scoring=scoring,
              verbose=5)#,n_jobs=8)
    print y
    rfecv.fit(X, y)
    print(' Optimal number of features: %d' % rfecv.n_features_)
    sel_features = [f for f, s in zip(X.columns, rfecv.support_) if s]
    
    temp = pd.DataFrame()
    temp['rfecv_score']=rfecv.grid_scores_
    temp['index']=range(1, len(rfecv.grid_scores_) + 1)
    temp.to_csv(os.path.join(path_save,'rfecv_score_'+rfc_name+'_'+str(index)+'.txt'))
    
    file_res_name = os.path.join(path_save, 'RFECV_'+scoring+'_'+rfc_name+'_'+str(index)+'.txt')
    print file_res_name
    open(file_res_name,'w').write('\n'.join(sel_features))
    

    
def optimize(X='', y='', scoring = 'roc_auc', selected_features=[], path_save='', index=''):
     
    X = X[selected_features]
    print y.shape
    print X.shape

    def to_maximaize(
                           max_features,
                           min_samples_leaf ,
                           min_weight_fraction_leaf,
                           min_samples_split,
                           ):
        
        
        
        rfc = ensemble.ExtraTreesClassifier(n_estimators=5,
                                     n_jobs=8,
                                     class_weight = 'balanced',
                                     min_samples_split= round(min_samples_split,1),
                                     max_features=round(max_features,1),    
                                     min_samples_leaf=round(min_samples_leaf,1),
                                     min_weight_fraction_leaf=round(min_weight_fraction_leaf,1)
                                     )
        
        parz_res = []
        for temp_round in range(5): 
            cv_result =  cross_val_score(rfc, X, y, scoring=scoring, cv=5)
            parz_res.append(cv_result.mean())
            
        return  np.mean(parz_res)
        
    xgbBO = BayesianOptimization(to_maximaize, {'max_features': (0.1, 1),
                                                'min_samples_leaf': (0.1, 0.5),
                                                'min_weight_fraction_leaf': (0.1, 0.5),
                                                'min_samples_split':(0.1, 0.5),

                                                })    
    xgbBO.maximize(init_points=10, n_iter=10)

    res = xgbBO.res['max']['max_params']
    print res


    rfc =  ensemble.ExtraTreesClassifier(n_estimators=10,#n_estimators=int(res['n_estimators']),
                                 n_jobs=8,
                                 class_weight = 'balanced',
                                 min_samples_split= round(res['min_samples_split'],1),
                                 max_features=round(res['max_features'],1),
                                 min_samples_leaf=round(res['min_samples_leaf'],1),
                                 min_weight_fraction_leaf=round(res['min_weight_fraction_leaf'], 1)
                                 )

    
    file_res_name = os.path.join(path_save,'scores_res_ExtraTreesClassifier_'+scoring+'_'+str(index)+'_opt.txt')            
    scores_res = open(file_res_name, 'w')
    for score in ['adjusted_rand_score','accuracy','f1','f1_weighted','precision_weighted','recall_weighted','roc_auc']:
        cv_result =  cross_val_score(rfc, X[selected_features], y, scoring=score, cv=5)
        print score, np.mean(cv_result)
        scores_res.write(str(score)+' ' +str(cv_result.mean())+' '+str(cv_result.std())+' '+str(cv_result)+'\n')
    
    scores_res.close()
    rfc.fit(X,y)
    joblib.dump(rfc, os.path.join(path_save,'ExtraTreesClassifier_optimized_'+scoring+'_'+str(index)+'.pkl')) 

 
           
#http://sebastianraschka.com/Articles/2014_ensemble_classifier.html        
#http://stackoverflow.com/questions/21506128/best-way-to-combine-probabilistic-classifiers-in-scikit-learn  

    
if __name__ == '__main__':
    
    path_save = os.path.join('optimized','etc')
    os.makedirs(path_save)
    
    for index in np.arange(1,11,1):
        index=1
        start_df = 'predictor_dataset/'+'in_data_'+str(index)+'.csv'
        start_df = pd.DataFrame.from_csv(start_df,index_col=[0,1])
        #print start_df.head()
        start_df = start_df.fillna(-1)
        X = start_df.iloc[:,0:-1]
        #print X.head()
        y = start_df['class']
        rfc = ensemble.ExtraTreesClassifier(n_estimators=10)
        
        rfc_name = get_predictor_name(rfc)
        scoring = 'f1_weighted'
        
        make_RFECV(scoring = scoring, rfc=rfc, X=X, y=y, index=index, path_save=path_save)
        selected_features = os.path.join(path_save,'RFECV_'+scoring+'_'+rfc_name+'_'+str(index)+'.txt')
        selected_features = [strip(n) for n in open(selected_features).readlines()]
        print selected_features[0:10]
        
        optimize(X=X, y=y, scoring = 'roc_auc', selected_features=selected_features, index=index, path_save=path_save)
        break                   
                    
    
