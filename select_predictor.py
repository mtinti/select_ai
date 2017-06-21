# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 10:19:23 2017

@author: mtinti-x
"""
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import ensemble,tree,svm,linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import neural_network
from sklearn.feature_selection import SelectPercentile, f_classif
from numpy import inf
from utility import plot_score, plot_feature_importance, get_predictor_name

#possible scores
#'accuracy', 'adjusted_rand_score', 'average_precision', 
#'f1', 'f1_macro', 'f1_micro',  'f1_weighted', 'neg_log_loss', 
#'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 
#'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
#'r2', 'recall', 'recall_macro', 'recall_micro',  'recall_weighted', 'roc_auc'


def find_feature_importance(rfc='', X='',y='', index=0, type_predictor = 'tree', path_save_files=''):
    print 
    #rfc = RandomForestClassifier(n_estimators=100, n_jobs=3)
    rfc_name = get_predictor_name(rfc)
    rfc.fit(X,y)
    if type_predictor == 'tree':
        importances = rfc.feature_importances_
    if type_predictor == 'logistic':
        importances = rfc.named_steps['predictor'].coef_ [0]
    if type_predictor == 'svm':
        selector = SelectPercentile(f_classif, percentile=10)  
        selector.fit(X, y)
        importances = -np.log10(selector.pvalues_)
        importances[importances == inf] = importances.max()+1000
    indices = np.argsort(importances)[::-1]
    res_df = pd.DataFrame()
    f_indexes = []
    f_names = []
    f_scores = []
    for f in range(X.shape[1]):
        f_indexes.append(f + 1)
        f_names.append(X.columns[indices[f]])
        f_scores.append( importances[indices[f]] )
    res_df['indexes']=f_indexes
    res_df['names']=f_names
    res_df['scores']=f_scores
    res_df.head()
    file_res_name = os.path.join(path_save_files,'FeatureImportance_'+rfc_name+'_'+str(index)+'.csv')
    res_df.to_csv(file_res_name)


def make_simple_predictor(rfc, X, y, index, path_save_files=''):
    print 
    rfc_name = get_predictor_name(rfc)
    file_res_name = os.path.join(path_save_files,'base_scores_'+rfc_name+'_'+str(index)+'.txt')
    scores_res = open(file_res_name,'w')
    for score in  ['accuracy','f1','f1_weighted','precision_weighted','recall_weighted','roc_auc']:
        cv_result =  cross_val_score(rfc, X, y, scoring=score, cv=5, pre_dispatch=5)
        scores_res.write(str(score)+' '+str(np.mean(cv_result))+' '+str(np.std(cv_result))+' '+','.join([str(n) for n in cv_result])+'\n')
        print index, rfc_name, score, np.mean(cv_result), np.std(cv_result)
    scores_res.close()
    return 1

def get_scores():
    res = {}
    for predictor in os.listdir('base_estimators'):
        if not predictor.startswith('.'):
            for file_name in os.listdir(os.path.join('base_estimators',predictor)):
                if 'base_scores_' in file_name:
                    in_file_name = os.path.join('base_estimators', predictor, file_name)
                    temp_df = pd.read_table(in_file_name,
                                                    sep=' ',
                                                    header = None,
                                                    index_col =0)
                    temp_df.index.names = ['score_type'] 
                    temp_df.columns = ['mean','std','values']
                    if predictor in res:
                        res[predictor]+=[ temp_df ]
                    else:
                        res[predictor]=[ temp_df ]
    return res    
   
X_scaler = StandardScaler()    
pipe_1 = Pipeline(  steps=[ ('scaling',X_scaler), ('predictor', linear_model.LogisticRegression(random_state=1)) ])       
pipe_2 = Pipeline(  steps=[ ('scaling',X_scaler), ('predictor', svm.SVC(probability=True)) ])    
pipe_3 = Pipeline(  steps=[ ('scaling',X_scaler), ('predictor', neural_network.MLPClassifier()) ]) 
   

predictor_list = [
                tree.DecisionTreeClassifier(),
                ensemble.ExtraTreesClassifier(n_estimators=50),
                ensemble.RandomForestClassifier(n_estimators=50),
                pipe_1,
                pipe_2,
                pipe_3
                ]       


if __name__ == '__main__':
    '''  
    for rfc in predictor_list[0:3]:
        rfc_name = get_predictor_name(rfc)
        path_save_files = os.path.join('base_estimators', rfc_name)
        os.makedirs(path_save_files)
        for index in range(1,11,1):
            start_df = os.path.join('predictor_dataset','in_data_'+str(index)+'.csv')
            start_df = pd.DataFrame.from_csv(start_df, index_col=[0,1])
            #print start_df.head()
            start_df = start_df.fillna(-1)
            X = start_df.iloc[:,0:-1]
            #print X.head()
            y = start_df['class']
            find_feature_importance(rfc, X, y, index, 'tree', path_save_files)
            make_simple_predictor(rfc, X, y, index, path_save_files)
        
    
    
    
    
    rfc = predictor_list[3]
    rfc_name = get_predictor_name(rfc)
    path_save_files = os.path.join('base_estimators', rfc_name)
    os.makedirs(path_save_files)    
    for index in range(1,11,1):
        start_df = os.path.join('predictor_dataset','in_data_'+str(index)+'.csv')
        start_df = pd.DataFrame.from_csv(start_df,index_col=[0,1])
        #print start_df.head()
        start_df = start_df.fillna(-1)
        X = start_df.iloc[:,0:-1]
        #print X.head()
        y = start_df['class']
        find_feature_importance(rfc, X, y, index, 'logistic', path_save_files)
        make_simple_predictor(rfc, X, y, index, path_save_files)
        
        
    for rfc in predictor_list[4:]:
        rfc_name = get_predictor_name(rfc)
        path_save_files = os.path.join('base_estimators', rfc_name)
        os.makedirs(path_save_files)
        for index in range(1,11,1):
            start_df = 'predictor_dataset/'+'in_data_'+str(index)+'.csv'
            start_df = pd.DataFrame.from_csv(start_df,index_col=[0,1])
            #print start_df.head()
            start_df = start_df.fillna(-1)
            X = start_df.iloc[:,0:-1]
            #print X.head()
            y = start_df['class']
            find_feature_importance(rfc, X, y, index, 'svm', path_save_files)
            make_simple_predictor(rfc, X, y, index, path_save_files)            
             
    '''
    plot_feature_importance('base_estimators/RandomForestClassifier')
    plot_feature_importance('base_estimators/ExtraTreesClassifier') 
    plot_feature_importance('base_estimators/Pipeline_SVC')     
    plot_feature_importance('base_estimators/Pipeline_MLPClassifier')
    plot_feature_importance('base_estimators/Pipeline_LogisticRegression')
    
    dict_df = get_scores()
    plot_score(in_score='roc_auc', dict_df=dict_df)
    plot_score(in_score='f1', dict_df=dict_df)
    plot_score(in_score='precision_weighted', dict_df=dict_df)    
    plot_score(in_score='recall_weighted', dict_df=dict_df) 
