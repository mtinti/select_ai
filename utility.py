

import pandas as pd
from string import strip
import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'                
import os
       
def plot_score(in_score='',dict_df={}):
    res_df = pd.DataFrame()
    for predictor in dict_df:
        parz_res = []
        df_list = dict_df[predictor]
        for temp_df in df_list:
            values = [float(strip(n)) for n in temp_df['values'][in_score].split(',')]
            parz_res+=values
        res_df[predictor]=parz_res
    fig,ax = plt.subplots()
    res_df.plot(kind='box',ax=ax) 
    ax.set_xticklabels(res_df.columns, rotation=-40, ha='left')
    ax.set_ylim(0.5,1.05)
    ax.set_title(in_score)
    plt.show()
    

def plot_feature_importance(predictor_path):    
    series_list = []
    col_name = []
    for file_name in os.listdir(predictor_path):
        if 'FeatureImportance_' in file_name:
            temp_df = pd.DataFrame.from_csv(predictor_path+'/'+file_name)
            temp_df = temp_df.set_index('names')
            series_list.append(temp_df['scores'])
            col_name.append(file_name.split('_')[-1].split('.')[0])
    
    res_df = pd.concat(series_list,axis=1)
    res_df.columns = ['round_'+n for n in col_name]
    res_df['median'] = res_df.median(axis=1)
    res_df =res_df.sort_values(by='median',ascending=True)    
    res_df = res_df.iloc[-60:,:]      
    res_df = res_df.T
    fig,ax = plt.subplots(figsize=(4,24))
    res_df.plot(kind='box',ax=ax,vert=False) 
    #ax.set_xticklabels(res_df.columns, rotation=-40, ha='left')
    #ax.set_ylim(0,1.1)
    ax.set_title('feature importance '+predictor_path.split('/')[-1])
    plt.show()
    
    
def get_predictor_name(rfc):
    rfc_name = str(rfc).split('(')[0]
    if rfc_name == 'Pipeline':
        rfc_name=rfc_name+'_'+str(rfc.named_steps['predictor']).split('(')[0]
    return rfc_name