import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import os

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

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
## Descriptives
##########################################
non_financial_variables = get_non_financial_variables()

columns = pd.MultiIndex.from_tuples(
    [(None,'All','Imputed'),
    (None,'All','Mean'),(None,'All','Median'),(None,'All','Std'),
    (None,'Non-bankrupted','Mean'),(None,'Non-bankrupted','Median'),(None,'Non-bankrupted','Std'),
    ('Bankrupted','1 year horizon','Mean'),('Bankrupted','1 year horizon','Median'),('Bankrupted','1 year horizon','Std'),
    ('Bankrupted','2 years horizon','Mean'),('Bankrupted','2 years horizon','Median'),('Bankrupted','2 years horizon','Std'),
    ('Bankrupted','3 years horizon','Mean'),('Bankrupted','3 years horizon','Median'),('Bankrupted','3 years horizon','Std'),
    ]
)
df_descriptives = pd.DataFrame(index=non_financial_variables,columns=columns)

for var in non_financial_variables:
    temp = data[var].astype(float)
    df_descriptives.at[var,(None,'All','Imputed')] = data[var+'-missing'].mean()
    df_descriptives.at[var,(None,'All','Mean')]      = temp.mean()
    df_descriptives.at[var,(None,'All','Median')]    = temp.median()
    df_descriptives.at[var,(None,'All','Std')]       = temp.std(ddof=0)

    ind = (data['bankrupt_1']==0) & (data['bankrupt_2']==0) & (data['bankrupt_3']==0)
    temp = data.loc[ind,var].astype(float)
    df_descriptives.at[var,(None,'Non-bankrupted','Mean')]      = temp.mean()
    df_descriptives.at[var,(None,'Non-bankrupted','Median')]    = temp.median()
    df_descriptives.at[var,(None,'Non-bankrupted','Std')]       = temp.std(ddof=0)

    ind = data['bankrupt_1']==1
    temp = data.loc[ind,var].astype(float)
    df_descriptives.at[var,('Bankrupted','1 year horizon','Mean')]      = temp.mean()
    df_descriptives.at[var,('Bankrupted','1 year horizon','Median')]    = temp.median()
    df_descriptives.at[var,('Bankrupted','1 year horizon','Std')]       = temp.std(ddof=0)

    ind = data['bankrupt_2']==1
    temp = data.loc[ind,var].astype(float)
    df_descriptives.at[var,('Bankrupted','2 years horizon','Mean')]      = temp.mean()
    df_descriptives.at[var,('Bankrupted','2 years horizon','Median')]    = temp.median()
    df_descriptives.at[var,('Bankrupted','2 years horizon','Std')]       = temp.std(ddof=0)

    ind = data['bankrupt_3']==1
    temp = data.loc[ind,var].astype(float)
    df_descriptives.at[var,('Bankrupted','3 years horizon','Mean')]      = temp.mean()
    df_descriptives.at[var,('Bankrupted','3 years horizon','Median')]    = temp.median()
    df_descriptives.at[var,('Bankrupted','3 years horizon','Std')]       = temp.std(ddof=0)

# Saving
df_descriptives.to_excel('descriptives/descriptives.xlsx')

##########################################
## Number of observations
##########################################
unique_regnaar = list(np.sort(data['regnaar'].unique()))

bankrupt_columns = [
    'bankrupt_1',
    'bankrupt_2',
    'bankrupt_3',
]
columns = [
    'Number of observations',
    'Number of companies',
]+bankrupt_columns
df_num_observations = pd.DataFrame(index=unique_regnaar+['Total'],columns=columns)

for regnaar in unique_regnaar:
    temp = data[data['regnaar']==regnaar]
    df_num_observations.at[regnaar,'Number of observations']    = temp.shape[0]
    df_num_observations.at[regnaar,'Number of companies']       = len(temp.orgnr.unique())
    for i in bankrupt_columns:
        df_num_observations.at[regnaar,i] = temp[i].mean()
df_num_observations.at['Total','Number of observations']    = data.shape[0]
df_num_observations.at['Total','Number of companies']       = len(data.orgnr.unique())
for i in bankrupt_columns:
    df_num_observations.at['Total',i] = data[i].mean()

# Saving
df_num_observations.to_excel('descriptives/number_of_observations.xlsx')

##########################################
## Number of observations per industry level 2
##########################################
unique_industry = list(np.sort(data['naeringskoder_level_1'].unique()))

bankrupt_columns = [
    'bankrupt_1',
    'bankrupt_2',
    'bankrupt_3',
]
columns = [
    'Number of observations',
    'Number of companies',
]+bankrupt_columns
df_num_observations = pd.DataFrame(index=unique_industry+['Total'],columns=columns)

for industry in unique_industry:
    temp = data[data['naeringskoder_level_1']==industry]
    df_num_observations.at[industry,'Number of observations']    = temp.shape[0]
    df_num_observations.at[industry,'Number of companies']       = len(temp.orgnr.unique())
    for i in bankrupt_columns:
        df_num_observations.at[industry,i] = temp[i].mean()
df_num_observations.at['Total','Number of observations']    = data.shape[0]
df_num_observations.at['Total','Number of companies']       = len(data.orgnr.unique())
for i in bankrupt_columns:
    df_num_observations.at['Total',i] = data[i].mean()

#
df_num_observations = df_num_observations.reset_index()
df_num_observations = df_num_observations.rename(columns={'index': 'temp2'}, inplace=False)

industry_codes_to_names = pd.read_csv('../data_other/industry_codes_to_names_level_1.csv',sep=';',low_memory=False)

for i in range(df_num_observations.shape[0]-1):
    mask = industry_codes_to_names['code']==df_num_observations.at[i,'temp2']
    df_num_observations.at[i,'temp2'] = industry_codes_to_names.loc[mask,'shortName'].iat[0]

rename_dict = {
    'temp2': 'Level 1',
}
df_num_observations = df_num_observations.rename(columns=rename_dict, inplace=False)


# Saving
df_num_observations.to_excel('descriptives/number_of_observations_per_industry_level_1.xlsx',index=False)

##########################################
## Number of observations per industry level 2
##########################################
unique_industry = list(np.sort(data['naeringskoder_level_2'].unique()))

bankrupt_columns = [
    'bankrupt_1',
    'bankrupt_2',
    'bankrupt_3',
]
columns = [
    'Number of observations',
    'Number of companies',
]+bankrupt_columns
df_num_observations = pd.DataFrame(index=unique_industry+['Total'],columns=columns)

for industry in unique_industry:
    temp = data[data['naeringskoder_level_2']==industry]
    df_num_observations.at[industry,'Number of observations']    = temp.shape[0]
    df_num_observations.at[industry,'Number of companies']       = len(temp.orgnr.unique())
    for i in bankrupt_columns:
        df_num_observations.at[industry,i] = temp[i].mean()
df_num_observations.at['Total','Number of observations']    = data.shape[0]
df_num_observations.at['Total','Number of companies']       = len(data.orgnr.unique())
for i in bankrupt_columns:
    df_num_observations.at['Total',i] = data[i].mean()

#
df_num_observations = df_num_observations.reset_index()
df_num_observations.insert(0, 'temp1', None)
df_num_observations = df_num_observations.rename(columns={'index': 'temp2'}, inplace=False)

industry_codes_to_names = pd.read_csv('../data_other/industry_codes_to_names.csv',sep=';',low_memory=False)
for i in range(df_num_observations.shape[0]-1):
    mask = industry_codes_to_names['code']==df_num_observations.at[i,'temp2']
    df_num_observations.at[i,'temp2'] = industry_codes_to_names.loc[mask,'shortName'].iat[0]
    df_num_observations.at[i,'temp1'] = industry_codes_to_names.loc[mask,'parentCode'].iat[0]

rename_dict = {
    'temp1': 'Level 1',
    'temp2': 'Level 2',
}
df_num_observations = df_num_observations.rename(columns=rename_dict, inplace=False)

# Saving
df_num_observations.to_excel('descriptives/number_of_observations_per_industry_level_2.xlsx',index=False)
