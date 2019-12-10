# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:50:51 2019

@author: laramos
"""

#Creating nice plots
            
import seaborn as sns
import matplotlib.pyplot as plt

frame['mrs']=Y_mrs

def plot_box(var):

    
sum_poor=list  ()             
sum_good=list()
sum_nan=list()
var = 'rr_syst'

for i in range(0,frame.shape[0]):
    
    if np.isnan(frame['mrs'].iloc[i]):
        sum_nan.append(frame[var].iloc[i])
    else:
        if frame['mrs'].iloc[i]>=5:
            sum_poor.append(frame[var].iloc[i])
        else:
            sum_good.append(frame[var].iloc[i])

df_poor = pd.DataFrame(sum_poor,columns=['mRS 5-6'])
df_good = pd.DataFrame(sum_good,columns=['mRS 0-4'])


df_plot = pd.concat([df_poor,df_good],axis=1)

sns.set(font_scale = 2)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(df_plot),orient="h",color='b').set_ylabel('time to groin',fontsize=20)


sns.set(font_scale = 2)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(sum_poor),orient="h",color='b').set_ylabel('AGE 5-6',fontsize=20)
sns.set(font_scale = 2)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(sum_good),orient="h",color='b').set_ylabel('AGE 0-4',fontsize=20)



sns.set(font_scale = 2)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(frame['age']),orient="h",color='b').set_ylabel('Age',fontsize=20)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(frame['ASPECTS_BL']),orient="h",color= 'r').set_ylabel('ASPECTS',fontsize=20)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(frame['NIHSS_BL']),orient="h",color = 'g').set_ylabel('NIHSS',fontsize=20)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(frame['togroin']),orient="h",color = 'k' ).set_ylabel('Time to Groin',fontsize=20)
plt.figure(figsize=(15, 2))
ax = sns.boxplot(data=(frame['rr_syst']),orient="h",color ='c' ).set_ylabel('Systolic Blood Pressure',fontsize=20)


import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.3

spec = np.array([0.94,0.93,0.96,0.96,0.96])

ci = np.array([[0.93,0.96],[0.89,0.96],[0.95,0.97],[0.94,0.97],[0.95,0.98]])

yerr = np.c_[spec-ci[:,0],ci[:,1]-spec ].T

plt.bar(range(len(spec)), spec, yerr=yerr)
plt.xticks(range(len(spec)))
plt.show()

y_r = [spec[i] - ci[i][1] for i in range(len(ci))]

plt.bar(range(len(spec)), spec, yerr=y_r, alpha=0.2, align='center')
plt.xticks(range(len(spec)), [str(year) for year in range(1992, 1996)])
plt.show()

r1 = np.arange(len(spec))
r2 = [x + barWidth for x in r1]

plt.bar(r1, spec, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yerr, capsize=7, label='poacee')