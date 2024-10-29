import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

import os

from imblearn.over_sampling import SMOTE

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *

############################################################
## Load data
############################################################
data = pd.read_csv('../data_4_imputed/data_imputed.csv',sep=';',low_memory=False)

folder_name = '../data_5_sampled'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

variables_for_sampling = get_variables_altman_1968() 
variables_for_sampling += get_variables_altman_and_sabato_2007()
variables_for_sampling += get_variables_paraschiv_2021() 
variables_for_sampling += get_non_financial_variables()

unique_regnaar = data['regnaar'].unique()

for bankrupt_var in ['bankrupt_1', 'bankrupt_2', 'bankrupt_3']:

    for regnaar in tqdm(unique_regnaar):
        
        data_regnaar = data[(data['regnaar'] == regnaar)]
        X = data_regnaar[variables_for_sampling]
        y = data_regnaar[bankrupt_var]
   
        # SMOTE, to oversample the minority class
        smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Create a DataFrame with the resampled data
        X_resampled[bankrupt_var] = y_resampled
        X_resampled['regnaar'] = regnaar

        if regnaar == unique_regnaar[0]:
            data_sampled = X_resampled.copy()
        else:
            data_sampled = pd.concat([data_sampled, X_resampled], axis=0)

        filename = '{}/data_sampled_{}.csv'.format(folder_name, bankrupt_var)

    data_sampled.to_csv(filename, index=False, sep=';')

