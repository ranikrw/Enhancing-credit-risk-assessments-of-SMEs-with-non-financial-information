import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

from sklearn.impute import KNNImputer
import os
from sklearn.preprocessing import StandardScaler

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *

############################################################
## Load data
############################################################
data = pd.read_csv('../data_3_for_analysis/data_with_missing.csv',sep=';',low_memory=False)

############################################################
## Impute per accounting year
############################################################
var_use = get_variables_paraschiv_2021()

# Checking that variables used in the vector space
# for imputation do not have missing values
if np.sum(np.sum(pd.isnull(data[var_use])))!=0:
    print('ERROR')

variables_non_financial = get_non_financial_variables()

data_imputed = pd.DataFrame()
for regnaar in np.sort(data['regnaar'].unique()): # Per accounting year
    print('Imputing accounting year {}:'.format(regnaar))
    
    data_regnaar = data[data['regnaar']==regnaar].reset_index(drop=True)

    # Creating vector space used for the kNN-algorithm
    data_var_use = data_regnaar[var_use]

    # Standardizing the vector space to a mean of 0 and 
    # standard deviation of 1 for each variable, respectively
    data_var_use = pd.DataFrame(StandardScaler().fit_transform(data_var_use),columns=data_var_use.columns)

    for var in tqdm(variables_non_financial):
        data_regnaar[var+'-missing'] = data_regnaar[var].isnull().astype(int)
        if data_regnaar[var].isnull().sum()!=0: # Impute only if there are missing values
            data_for_imputation = pd.concat([data_regnaar[var],data_var_use],axis=1)
            if len(data_regnaar[var].unique())<=3: # If dummy
                imputer = KNNImputer(n_neighbors=1)
            else: # If not dummy
                imputer = KNNImputer(n_neighbors=3,weights='distance')
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation)) # Impute
            data_regnaar[var] = temp[0].copy() # First column is the one imputed, so replacing this one in data

    data_imputed = pd.concat([data_imputed, data_regnaar],axis=0)

# Reset index
data_imputed = data_imputed.reset_index(drop=True)

# Checking
if data_imputed.shape[0]!=data.shape[0]:
    print('ERROR when imputing')
if data_imputed.shape[1]!=(data.shape[1]+len(variables_non_financial)):
    print('ERROR when imputing')
if np.sum(data_imputed['orgnr'])!=np.sum(data['orgnr']):
    print('ERROR when imputing')
if np.sum(data_imputed['regnaar'])!=np.sum(data['regnaar']):
    print('ERROR when imputing')

temp = (np.sum(data_imputed['log(age in years)'])-np.sum(data['log(age in years)']))/np.sum(data['log(age in years)'])
if temp>1e-09:
    print('ERROR when imputing')
temp = (np.sum(data_imputed['EBIT / total assets'])-np.sum(data['EBIT / total assets']))/np.sum(data['EBIT / total assets'])
if temp>1e-09:
    print('ERROR when imputing')

############################################################
## Save to file
############################################################
folder_name = '../data_4_imputed'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

data_imputed.to_csv(folder_name+'/data_imputed.csv',index=False,sep=';')


