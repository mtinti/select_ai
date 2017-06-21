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
from utility import get_predictor_name

#avaiable scores

# 'adjusted_rand_score', 'average_precision', 
# 'f1', 'f1_macro', 'f1_micro',  'f1_weighted', 'neg_log_loss', 
#'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 
#'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
# 'r2', 'recall', 'recall_macro', 'recall_micro',  'recall_weighted', 'roc_auc'


path_save_files = 'optimized/etc/'

def make_RFECV(scoring = '', rfc='', X='', y='', index=0):
    rfc_name = str(rfc).split('(')[0]

    rfecv = RFECV(estimator=rfc,
              step=1,
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
    temp.to_csv(path_save_files+'rfecv_score_'+rfc_name+'_'+str(index)+'.txt')
    open(path_save_files+'RFECV_'+scoring+'_'+rfc_name+'_'+str(index)+'.txt','w').write('\n'.join(sel_features))
    

    
def optimize(X='', y='', scoring = 'roc_auc', selected_features=[]):
      
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
    xgbBO.maximize(init_points=20, n_iter=150)

    res = xgbBO.res['max']['max_params']
    print res


    rfc =  ensemble.ExtraTreesClassifier(n_estimators=500,#n_estimators=int(res['n_estimators']),
                                 n_jobs=8,
                                 class_weight = 'balanced',
                                 min_samples_split= round(res['min_samples_split'],1),
                                 max_features=round(res['max_features'],1),
                                 min_samples_leaf=round(res['min_samples_leaf'],1),
                                 min_weight_fraction_leaf=round(res['min_weight_fraction_leaf'], 1)
                                 )

    
                  
    scores_res = open(pat_save_files+'scores_res_ExtraTreesClassifier_'+scoring+'_opt.txt','w')
    for score in ['adjusted_rand_score','accuracy','f1','f1_weighted','precision_weighted','recall_weighted','roc_auc']:
        cv_result =  cross_val_score(rfc, X[selected_features], y, scoring=score, cv=5)
        print score, np.mean(cv_result)
        scores_res.write(str(score)+' ' +str(cv_result.mean())+' '+str(cv_result.std())+' '+str(cv_result)+'\n')
    
    scores_res.close()
    rfc.fit(X,y)
    joblib.dump(rfc, 'ExtraTreesClassifier_optimized_'+scoring+'.pkl') 

 
           
#http://sebastianraschka.com/Articles/2014_ensemble_classifier.html        
#http://stackoverflow.com/questions/21506128/best-way-to-combine-probabilistic-classifiers-in-scikit-learn  
rfc_list = [
            #tree.DecisionTreeClassifier(),
            #tree.ExtraTreeClassifier(),
            #ensemble.AdaBoostClassifier(n_estimators=500),
            
            #ensemble.GradientBoostingClassifier(n_estimators=500),
            ensemble.RandomForestClassifier(n_estimators=500),
            #linear_model.LogisticRegression(random_state=1),
            #naive_bayes.GaussianNB()
            ]
    
    
    

if __name__ == '__main__':
    

    
    
    
    for index in np.arange(1,11,1):
        print index
        start_df = 'predictor_dataset/'+'in_data_'+str(index)+'.csv'
        start_df = pd.DataFrame.from_csv(start_df,index_col=[0,1])
        #print start_df.head()
        start_df = start_df.fillna(-1)
        X = start_df.iloc[:,0:-1]
        #print X.head()
        y = start_df['class']
        rfc = ensemble.ExtraTreesClassifier(n_estimators=50)
        make_RFECV(scoring = 'f1_weighted', rfc=rfc, X=X, y=y, index=index)
    
     
    start_df = 'predictor_dataset/'+'in_data_'+str(1)+'.csv'
    start_df = pd.DataFrame.from_csv(start_df,index_col=[0,1]) 
    rfc = ensemble.ExtraTreesClassifier(n_estimators=10)
    X = start_df.iloc[:,0:-1]
    y = start_df['class']
    optimize(X=X, y=y, scoring = 'roc_auc', selected_features=X.columns.values[0:10])                   
                    
    
