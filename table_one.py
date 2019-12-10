# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:48:47 2019

@author: laramos
"""

import pandas as pd
import numpy as np

var_list = dict()
#var_list['age'] = 'cont'
#var_list['sex'] = 'bin'
#var_list['togroin'] = 'cont'
#var_list['NIHSS_BL'] = 'cont'
var_list['rr_syst'] = 'cont'
#var_list['ivtrom'] = 'bin'
#var_list['ASPECTS_BL'] = [(0,4),(5,7),(8,10)]
#var_list['CBS_BL'] = 'cat'
#var_list['cbs_occlsegment_recoded'] = 'cat'
#var_list['collaterals'] = 'cat'
#var_list['premrs'] = 'cat'
#var_list['prev_dm'] = 'cat'
#var_list['glucose'] = 'cont'


#var_list['HYARS_E1_C1'] = 'bin'
#var_list['HYARS_E1_C1'] = 'occlsegment_c_short '

#Table_one(data,cols_o,Y_mrs>=5,var_list):

def Table_One(data,cols_o,label,var_list):
 
    frame = pd.DataFrame(data,columns=cols_o)   
    frame['Y']=label

    frame_poor = frame[frame['Y'] == 1]
    frame_good = frame[frame['Y'] == 0]
    cols_names = ['Names','Total','Positive','Negative']
    save_frame = pd.DataFrame(columns=cols_names)
    
    for var in var_list:
        
        if var_list[var] == 'bin':
                     
            tot = ("%.2f (%.2f)"%(frame[var].sum(),frame[var].sum()*100/frame.shape[0]))
            pos = ("%.2f (%.2f)"%(frame_poor[var].sum(),frame[var].sum()*100/frame_poor.shape[0]))
            neg = ("%.2f (%.2f)"%(frame_good[var].sum(),frame[var].sum()*100/frame_good.shape[0]))            
            to_app = pd.concat([pd.Series(var),pd.Series(tot), pd.Series(pos),pd.Series(neg)],axis=1)
            to_app.columns = cols_names
            save_frame = save_frame.append(to_app)
        else:
            
           if var_list[var] == 'cont': 
                name1 = var+' MEAN and STD'  
                tot = ("%.2f (%.2f)"%(frame[var].mean(),frame[var].std()))
                pos = ("%.2f (%.2f)"%(frame_poor[var].mean(),frame_poor[var].std()))
                neg = ("%.2f (%.2f)"%(frame_good[var].mean(),frame_good[var].std()))
                   
                name2 = var+' MEDIAN and IQR'    
                tot_2 = ("%.2f (%.2f - %.2f)"%(np.nanmedian(frame[var]),np.nanpercentile(frame[var], 75, interpolation='higher'),
                                                                 np.nanpercentile(frame[var], 25, interpolation='lower')))
                pos_2 = ("%.2f (%.2f - %.2f)"%(np.nanmedian(frame_poor[var]),np.nanpercentile(frame_poor[var], 75, interpolation='higher'),
                                                                  np.nanpercentile(frame_poor[var], 25, interpolation='lower')))
                neg_2 = ("%.2f (%.2f - %.2f)"%(np.nanmedian(frame_good[var]),np.nanpercentile(frame_good[var], 75, interpolation='higher'),
                                                                 np.nanpercentile(frame_good[var], 25, interpolation='lower')))
                to_app = pd.concat([pd.Series(name1),pd.Series(tot), pd.Series(pos),pd.Series(neg)],axis=1)
                to_app.columns = cols_names
                to_app_2 = pd.concat([pd.Series(name2),pd.Series(tot_2), pd.Series(pos_2),pd.Series(neg_2)],axis=1)
                to_app_2.columns = cols_names
                save_frame = save_frame.append(to_app)
                save_frame = save_frame.append(to_app_2)
                   
           else:
                if var_list[var] == 'cat': 
                    
                    vals_f = frame[var].value_counts()
                    vals_f_p  = round(vals_f*100/(frame.shape[0]),2)
                    f_f = pd.concat([vals_f, vals_f_p],axis=1)
                    f_f.columns = ['a','b']
                    f_f["comb"] = f_f["a"].map(str) +" ("+ f_f["b"].map(str)+')'                    
                                        
                    vals_poor = frame_poor[var].value_counts()
                    vals_poor_p  = round(vals_poor*100/(frame_poor.shape[0]),2)
                    f_f_p = pd.concat([vals_poor, vals_poor_p],axis=1)
                    f_f_p.columns = ['a','b']
                    f_f_p["comb"] = f_f_p["a"].map(str) +" ("+ f_f_p["b"].map(str)+')'
                                       
                    vals_good = frame_good[var].value_counts()
                    vals_good_p  = round(vals_good*100/(frame_good.shape[0]),2)
                    f_f_g = pd.concat([vals_good, vals_good_p],axis=1)
                    f_f_g.columns = ['a','b']
                    f_f_g["comb"] = f_f_g["a"].map(str) +" ("+ f_f_g["b"].map(str)+')'
                    
                    nn =pd.concat([f_f["comb"],f_f_p["comb"],f_f_g["comb"]],axis=1)
                    ind = nn.index
                    name = list()
                    for i in ind:
                        name.append(var+' = '+str(ind[i]))
                    nn['title'] = pd.Series(name)
                    nn.columns = ['Total','Positive','Negative','Names']
                    save_frame = save_frame.append(nn)
                else:
                    if type(var_list[var])==list:
                        max_g = len(var_list[var])
                        interval = var_list[var]
                        arr = np.zeros(max_g)
                        tot_f = list()                       
                        tot_p = list()                       
                        tot_g = list()                      
                        name = list()
                        for i in range(0,max_g):
                            name.append(var+": "+str(interval[i][0])+"-"+str(interval[i][1]))
                            vf = frame[var].between(interval[i][0], interval[i][1], inclusive=True).sum()
                            vp  = round(vf*100/(frame.shape[0]),2)
                            tot_f.append(str(vf)+" ("+str(vp)+')')
                            
                            vf = frame_poor[var].between(interval[i][0], interval[i][1], inclusive=True).sum()
                            vp  = round(vf*100/(frame.shape[0]),2)
                            tot_p.append(str(vf)+" ("+str(vp)+')')
                            
                            vf = frame_good[var].between(interval[i][0], interval[i][1], inclusive=True).sum()
                            vp  = round(vf*100/(frame.shape[0]),2)
                            tot_g.append(str(vf)+" ("+str(vp)+')')

                        nn = pd.concat([pd.Series(name),pd.Series(tot_f), pd.Series(tot_p),pd.Series(tot_g)],axis=1)
                        nn.columns = cols_names
                        save_frame = save_frame.append(nn)


#p_values
                        
#f=np.concatenate((X_train_imp,X_test_imp),axis=0)
#f=pd.DataFrame(f,columns=cols)
#y = np.concatenate((y_train.reshape(-1,1),y_test.reshape(-1,1)),axis=0).reshape(-1)
#import statsmodels.discrete.discrete_model as sm
#logit = sm.Logit(y, f)
#result = logit.fit()
#result.summary()
#
#
#
#
#f=pd.DataFrame(X_train_imp,columns=cols)
#import statsmodels.discrete.discrete_model as sm
#logit = sm.Logit(y_train, f)
#result = logit.fit()
#result.summary()

                        