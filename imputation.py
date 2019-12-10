# -*- coding: utf-8 -*-
"""

This code contains functions for imputation. Train and test data are imputed separately to prevent data leakage.


@author: laramos
"""
import numpy as np
from missingpy import KNNImputer,MissForest

def Impute_Data_MICE(X_train,y_train,X_test,y_test,n_imputations,vals_mask,cols,mrs):
    
    XY_incomplete = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_completed_train = []
    XY_completed_test = []
    
    for i in range(n_imputations):
        imputer = IterativeImputer(n_iter=n_imputations, sample_posterior=True, random_state=i,initial_strategy="mean",min_value=0)
        XY_completed_train.append(imputer.fit_transform(XY_incomplete))
        XY_completed_test.append(imputer.transform(np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)))
        
        if mrs:            
            X_train_imp=(XY_completed_train[i][:,0:data.shape[1]])
            y_train_imp=np.array(XY_completed_train[i][:,data.shape[1]]>=5,dtype="int16")
            X_test_imp=(XY_completed_test[i][:,0:data.shape[1]])
            y_test_imp=np.array(XY_completed_test[i][:,data.shape[1]]>=5,dtype="int16")
        else:
            X_train_imp=(XY_completed_train[i][:,0:data.shape[1]])
            y_train_imp=np.array(XY_completed_train[i][:,data.shape[1]]<3,dtype="int16")
            X_test_imp=(XY_completed_test[i][:,0:data.shape[1]])
            y_test_imp=np.array(XY_completed_test[i][:,data.shape[1]]<3,dtype="int16")
        
        for j in range(0,X_train_imp.shape[1]):
            if  var.iloc[j]['type']=='cat':
                X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
                X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
            else:
                X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=1)
                X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=1)
                
        #frame_train=pd.DataFrame(X_train_imp,columns=cols)
        #frame_test=pd.DataFrame(X_test_imp,columns=cols)
        
        #frame_train=pd.get_dummies(frame_train, columns=vals_mask)
        #frame_test=pd.get_dummies(frame_test, columns=vals_mask)
                    
        return(X_train_imp,y_train_imp,X_test_imp,y_test_imp)   
        
        

        
def Impute_Data_KNN(X_train,y_train,X_test,y_test,vals_mask,cols,data,var,min_vals,max_vals):
    
    XY_incomplete_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_incomplete_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)


    imputer = KNNImputer(n_neighbors=5)
    XY_completed_train = imputer.fit_transform(XY_incomplete_train)
    XY_completed_test = imputer.transform(XY_incomplete_test)
          
    X_train_imp=(XY_completed_train[:,0:data.shape[1]])
    y_train_imp_orig=np.array(XY_completed_train[:,data.shape[1]],dtype="int16")
    y_train_imp=np.array(XY_completed_train[:,data.shape[1]]>=5,dtype="int16")
    X_test_imp=(XY_completed_test[:,0:data.shape[1]])
    y_test_imp=np.array(XY_completed_test[:,data.shape[1]]>=5,dtype="int16")
    y_test_imp_orig=np.array(XY_completed_test[:,data.shape[1]],dtype="int16")

    
    
    for j in range(0,X_train_imp.shape[1]):
        if  var.iloc[j]['type']=='cat':
            X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
            X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
        else:
            X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=1)
            X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=1)
    
    #min_vals_imp=np.nanmin(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)
    #max_vals_imp=np.nanmax(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)  
                    
    return(X_train_imp,y_train_imp,X_test_imp,y_test_imp,y_train_imp_orig,y_test_imp_orig)           
   
def Impute_Data_RF(X_train,y_train,X_test,y_test,vals_mask,cols):
    
    XY_incomplete_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_incomplete_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)


    imputer = MissForest(random_state=1,n_jobs=-1)
    XY_completed_train = imputer.fit_transform(XY_incomplete_train)
    #min_vals_2=np.nanmin(XY_completed_train,axis=0)
    #max_vals_2=np.nanmax(XY_completed_train,axis=0)
    XY_completed_test = imputer.transform(XY_incomplete_test)
             
    X_train_imp=(XY_completed_train[:,0:data.shape[1]])
    y_train_imp=np.array(XY_completed_train[:,data.shape[1]]>=5,dtype="int16")
    X_test_imp=(XY_completed_test[:,0:data.shape[1]])
    y_test_imp=np.array(XY_completed_test[:,data.shape[1]]>=5,dtype="int16")

    
    for j in range(0,X_train_imp.shape[1]):
        if  var.iloc[j]['type']=='cat':
            X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
            X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
        else:
            X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=1)
            X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=1)
    
    #min_vals_imp=np.nanmin(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)
    #max_vals_imp=np.nanmax(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)  
                    
    return(X_train_imp,y_train_imp,X_test_imp,y_test_imp)          