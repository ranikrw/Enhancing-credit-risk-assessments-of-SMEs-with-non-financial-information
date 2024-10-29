import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import os

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *

##########################################
## Load data
##########################################
folder_name = '../data_4_imputed'
data = pd.read_csv(folder_name+'/data_imputed.csv',sep=';',low_memory=False)

##########################################
## Make folder for saving descriptives
##########################################
folder_name = 'descriptives'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

##########################################
## Correlation matrix
##########################################
from scipy.stats import pearsonr
def calculate_pvalues(corr_values):
    corr_values = corr_values.dropna()._get_numeric_data()
    corr_values_cols = pd.DataFrame(columns=corr_values.columns)
    pvalues = pd.DataFrame(None, index=corr_values.index, columns=corr_values.columns)
    for r in corr_values.columns:
        for c in corr_values.columns:
            pvalues.loc[r,c] = round(pearsonr(corr_values[r], corr_values[c])[1], 4)
    return pvalues

num_decimals = 2

financial_variables = get_variables_altman_1968()
financial_variables = financial_variables + get_variables_altman_and_sabato_2007()
financial_variables = financial_variables + get_variables_paraschiv_2021()
financial_variables = list(np.unique(financial_variables))
all_variables = get_non_financial_variables() + financial_variables


for non_fin_set in [
    'management',
    'board',
    'ownership',
]:
    if non_fin_set == 'management':
        non_fin_vars = get_non_financial_variables_management_structure()
    elif non_fin_set == 'board':
        non_fin_vars = get_non_financial_variables_board_structure()
    elif non_fin_set == 'ownership':
        non_fin_vars = get_non_financial_variables_ownership_structure()    
    else:
        raise Exception('Correlation matrix: wrongly defined non_fin_vars')

    variables = non_fin_vars + [item for item in all_variables if item not in non_fin_vars]

    temp = data[variables]

    corr_values = temp.astype(float).corr()
    p_values = calculate_pvalues(corr_values)
    corr_mtr = pd.DataFrame(None, index=corr_values.index, columns=non_fin_vars)
    corr_mtr_unformatted = pd.DataFrame(None, index=corr_values.index, columns=non_fin_vars)
    included_vars = []
    for c in corr_mtr.columns:
        for i in corr_mtr.index:
            if i != c:
                if i not in included_vars:
                    if c in non_fin_vars:
                        included_vars += [c]
                    val = corr_values.loc[i,c]
                    corr_mtr_unformatted.loc[i,c] = val
                    val = str(np.round(val,num_decimals))
                    if len(val.split('.')[1])<num_decimals:
                        val = val+'0'
                    if p_values.loc[i,c]<0.01:
                        corr_mtr.loc[i,c] = val+'***'
                    elif p_values.loc[i,c]<0.05:
                        corr_mtr.loc[i,c] = val+'**'
                    elif p_values.loc[i,c]<0.10:
                        corr_mtr.loc[i,c] = val+'*'
                    else:
                        corr_mtr.loc[i,c] = val
                    
    corr_mtr = corr_mtr.reset_index()
    corr_mtr.insert(0, 'temp1', None)
    corr_mtr = corr_mtr.rename(columns={'index': 'temp2'}, inplace=False)

    num = 0
    for c in non_fin_vars:
        num += 1
        corr_mtr.loc[corr_mtr['temp2']==c,'temp1'] = '({})'.format(num)
        corr_mtr = corr_mtr.rename(columns={c: '({})'.format(num)}, inplace=False)

    corr_mtr = corr_mtr.rename(columns={'temp1': None,'temp2': None}, inplace=False)

    # Saving
    corr_mtr.to_excel('descriptives/correlation_matrix_{}.xlsx'.format(non_fin_set),index=False)
    corr_mtr_unformatted.to_excel('descriptives/correlation_matrix_unformatted_{}.xlsx'.format(non_fin_set),index=True)

