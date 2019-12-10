# -*- coding: utf-8 -*-
"""

This code contains the main file for poor outcome prediction using the Mr Clean Registry dataset

@author: laramos
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV,LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder


from imblearn.under_sampling import RandomUnderSampler

cwd = os.getcwd()
os.chdir(cwd)

import methods as mt
import data_preprocessing as dp
import imputation as imp

#This class saves everything from the runs, it takes 2 arguments, itera: how many times you want to run the whole the whole analysis
# and split: how many folds you want to use

class Measures:       
    def __init__(self,itera,splits):
        self.clf_auc=np.zeros((itera,splits))
        self.clf_brier=np.zeros((itera,splits))
                        
        self.clf_f1_score=np.zeros((itera,splits))
        self.clf_sens=np.zeros((itera,splits))
        self.clf_spec=np.zeros((itera,splits))
        self.clf_ppv=np.zeros((itera,splits))
        self.clf_npv=np.zeros((itera,splits))
        
        self.sens_f1=np.zeros((itera,splits))
        self.spec_f1=np.zeros((itera,splits))
        self.f1_score_f1=np.zeros((itera,splits))
        self.clf_ppv_f1=np.zeros((itera,splits))
        self.clf_npv_f1=np.zeros((itera,splits))
        
        self.sens_spec=np.zeros((itera,splits))
        self.spec_spec=np.zeros((itera,splits))
        self.f1_score_spec=np.zeros((itera,splits))
        self.clf_ppv_spec=np.zeros((itera,splits))
        self.clf_npv_spec=np.zeros((itera,splits))
                    
        self.clf_thresholds=np.zeros((itera,splits))
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.frac_pos_rfc=np.zeros((itera,splits))
        self.run=False
        self.feat_imp=list() 
        self.probas=np.zeros(splits)
        self.preds=np.zeros(splits)

    
#path to the complete dataset    
path_data=r"\\amc.intra\users\L\laramos\home\Desktop\MrClean_Poor\data\data_complete.csv"
#path to the variables to the used
path_variables=r"\\amc.intra\users\L\laramos\home\Desktop\MrClean_Poor\data\Baseline_contscore_new.csv"

frame,cols_o,var,data,Y_mrs,Y_tici,data_img,vals_mask,miss,original_mrs,subj=dp.Clean_Data(path_data,path_variables)

print("Data Loaded")
min_vals=np.nanmin(data,axis=0)
max_vals=np.nanmax(data,axis=0)

#This is a parameter setup, N = no undersampling, Y with undersampling and W with Weights for class imbalance.
#if you dont have class imbalance set to N omly
undersample=['N','Y','W']
#imputation typ,e RF also available
imputation=['KNN']
optimizers=['roc_auc','f1','precision']

#How many times to run the whole pipeline? 
itera=1
#How many folds
splits=10
#How may inner folds
cv=5

mean_tprr = 0.0

rfc_m = Measures(itera,splits)
svm_m = Measures(itera,splits)
lr_m = Measures(itera,splits)
xgb_m = Measures(itera,splits)
nn_m = Measures(itera,splits)       

for i in imputation:
    for opt in optimizers:
        for und in undersample:
            path_results=(r"./feature_importance-16-10-"+i+"_opt-"+opt+"_und-"+und+"//")
      
            if not os.path.exists(path_results):
                os.makedirs(path_results)
                               
            for i in range(0,itera): 
                
                    start_pipeline = time.time()
                                
                    sk = KFold(n_splits=splits, shuffle=True, random_state=i)                
                    
                    for l, (train_index,test_index) in enumerate(sk.split(data, Y_mrs)):
                        
                        X_train, X_test = data[train_index,:], data[test_index,:]
                        y_train, y_test = Y_mrs[train_index], Y_mrs[test_index]   
                        
                        original_mrs_train = original_mrs[train_index]
                        original_mrs_test = original_mrs[test_index]
                        
                        #Saving the original mRS so we can create grotta bars
                        np.save(path_results+"original_mrs_train"+str(i)+str(l)+".npy",original_mrs_train)
                        np.save(path_results+"original_mrs_test"+str(i)+str(l)+".npy",original_mrs_test)
                        np.save(path_results+"id_train"+str(i)+str(l)+".npy",subj[train_index])
                        np.save(path_results+"id_test"+str(i)+str(l)+".npy",subj[test_index])
    
                        print("Imputing data! Iteration = ",l)                                        
                        
                        if i=='RF':
                            X_train_imp,y_train,X_test_imp,y_test=imp.Impute_Data_RF(X_train,y_train,X_test,y_test,vals_mask,cols_o)
                        else:
                            X_train_imp,y_train,X_test_imp,y_test,y_train_orig,y_test_orig=imp.Impute_Data_KNN(X_train,y_train,X_test,y_test,vals_mask,cols_o,data,var,min_vals,max_vals)
                                                   
                        X_train_imp,X_test_imp,cols = dp.Change_One_Hot(X_train_imp,X_test_imp,vals_mask,cols_o)
                        f_train = pd.DataFrame(X_train_imp,columns=cols)
                        f_test = pd.DataFrame(X_test_imp,columns=cols)
                        
                        #Normalizing the continuous variables
                        scaler = ColumnTransformer([('norm1', StandardScaler(), ['ASPECTS_BL', 'CBS_BL','NIHSS_BL', 'glucose','rr_syst','rr_dias','INR', 'trombo',
                                                 'crp','age','gcs','togroin', 'dtnt', 'dur_oct','dur_oer'])], remainder='passthrough')
                        
                        scaler = scaler.fit(f_train)    
                        X_train_imp=scaler.transform(f_train)
                        X_test_imp=scaler.transform(f_test)
                        
                        if und=='Y':
                            rus = RandomUnderSampler(random_state=1)
                            X_train_imp, y_train = rus.fit_resample(X_train_imp, y_train)   
    
    
                                            
                        class_rfc=mt.Pipeline(True,'RFC',X_train_imp,y_train,X_test_imp,y_test,l,cv,mean_tprr,rfc_m,path_results,opt,und,i)   
                        class_svm=mt.Pipeline(True,'SVM',X_train_imp,y_train,X_test_imp,y_test,l,cv,mean_tprr,svm_m,path_results,opt,und,i)   
                        class_lr=mt.Pipeline(True,'LR',X_train_imp,y_train,X_test_imp,y_test,l,cv,mean_tprr,lr_m,path_results,opt,und,i)
                        class_nn=mt.Pipeline(True,'NN',X_train_imp,y_train,X_test_imp,y_test,l,cv,mean_tprr,nn_m,path_results,opt,und,i) 
                        class_xgb=mt.Pipeline(True,'XGB',X_train_imp,y_train,X_test_imp,y_test,l,cv,mean_tprr,xgb_m,path_results,opt,und,i)  
                        end_pipeline = time.time()
                        print("Total time to process iteration: ",end_pipeline - start_pipeline)
                        
    
            final_m=[rfc_m,svm_m,lr_m,xgb_m,nn_m]
            final_m=[x for x in final_m if x.run != False]
            names=[class_rfc.name,class_svm.name,class_lr.name,class_xgb.name,class_nn.name]
            #names=[class_rfc.name,class_svm.name,class_lr.name,class_nn.name]
            names=[x for x in names if x != 'NONE'] 
            mt.Print_Results_Excel(final_m,splits,names,path_results,i)
                    
            mt.Save_fpr_tpr(path_results,names,final_m)
        


               
  
          





#%%


