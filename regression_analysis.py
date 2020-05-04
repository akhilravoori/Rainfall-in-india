#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 02:05:26 2020

@author: akhil
"""


import pandas as pd
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

rf_data=pd.read_csv('rainfall_area-wt_sd_1901-2015.csv')

st_data=rf_data.groupby('SUBDIVISION')
tn_data=st_data.get_group('TAMIL NADU')
rs_data=st_data.get_group('RAYALSEEMA')
cap_data=st_data.get_group('COASTAL ANDHRA PRADESH')

keys=rf_data.keys()
for key in keys:
    if key != 'ANNUAL' and key!='YEAR':
        tn_data=tn_data.drop(key,axis=1)
        rs_data=rs_data.drop(key,axis=1)
        cap_data=cap_data.drop(key,axis=1)

plt.figure()

plt.plot(tn_data.YEAR,tn_data.ANNUAL,label='tamil',alpha=0.75)
plt.plot(rs_data.YEAR,rs_data.ANNUAL,label='rayal',alpha=0.75)
plt.plot(cap_data.YEAR,cap_data.ANNUAL,label='coastal',alpha=0.75)

plt.show()

stats.pearsonr(tn_data.ANNUAL,rs_data.ANNUAL)
stats.pearsonr(tn_data.ANNUAL,cap_data.ANNUAL)
stats.pearsonr(cap_data.ANNUAL,rs_data.ANNUAL)

slr=regression.linear_model.OLS(np.array(rs_data.ANNUAL),sm.add_constant(np.array(cap_data.ANNUAL))).fit()
print("alpha and beta are:"+ str(slr.params[0])+' '+str(slr.params[1]))

prediction_rs=slr.params[0]+slr.params[1]*np.array(cap_data.ANNUAL)

plt.figure(figsize=(10,6))
plt.plot(rs_data.YEAR,rs_data.ANNUAL,label='actual',alpha=0.75)
plt.plot(rs_data.YEAR,prediction_rs,label='pred',alpha=0.75)

plt.show()

stats.pearsonr(prediction_rs,rs_data.ANNUAL)

slr2=regression.linear_model.OLS(np.array(rs_data.ANNUAL),sm.add_constant(np.array(tn_data.ANNUAL))).fit()
print("alpha and beta are:"+ str(slr.params[0])+' '+str(slr.params[1]))

prediction_rs2=slr2.params[0]+slr2.params[1]*np.array(tn_data.ANNUAL)

plt.figure(figsize=(10,6))
plt.plot(rs_data.YEAR,rs_data.ANNUAL,label='actual',alpha=0.75)
plt.plot(rs_data.YEAR,prediction_rs2,label='pred',alpha=0.75)

stats.pearsonr(prediction_rs2,rs_data.ANNUAL)

mlr=regression.linear_model.OLS(np.array(rs_data.ANNUAL),
                                sm.add_constant(np.column_stack((np.array(tn_data.ANNUAL),np.array(cap_data.ANNUAL))))).fit()

print("alpha and beta1 and beta2 are:"+ str(mlr.params[0])+' '+str(mlr.params[1])+' '+str(mlr.params[2]))                                                                    
print("alpha and beta are:"+ str(slr.params[0])+' '+str(slr.params[1]))

prediction_mlr=mlr.params[0]+mlr.params[1]*np.array(tn_data.ANNUAL)+mlr.params[2]*np.array(cap_data.ANNUAL)

plt.figure(figsize=(10,6))
plt.plot(rs_data.YEAR,rs_data.ANNUAL,label='actual',alpha=0.75)
plt.plot(rs_data.YEAR,prediction_mlr,label='pred',alpha=0.75)
plt.show()

stats.pearsonr(prediction_mlr,rs_data.ANNUAL)

rs_data['prediction_mlr']=prediction_mlr 
rs_data['prediction_slr1_tn']=prediction_rs
rs_data['prediction_slr2_cap']=prediction_rs2

rs_data




