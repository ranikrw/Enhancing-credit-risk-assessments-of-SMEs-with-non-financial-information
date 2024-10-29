import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

import os

##################################################################
##  Load data brreg                                             ##
##################################################################
print('-----------------------------------------')
print('Loading data:')
print('-----------------------------------------')
folder_name = '../../../datasett_aarsregnskaper/data4/'

years = [
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
]

for year in years:
    # LOADING DATA
    data_loaded = pd.read_csv(folder_name+str(year)+'.csv',sep=';',low_memory=False)

    # Adding all data together into data
    if year==years[0]:
        data_brreg = pd.DataFrame(columns=data_loaded.columns)
    data_brreg = pd.concat([data_brreg,data_loaded],axis=0)
    print('Imported for accounting year {}'.format(year))

# Reset index 
data_brreg = data_brreg.reset_index(drop=True)

# Double checking if all is unique
if data_brreg.shape[0] != data_brreg.drop_duplicates().shape[0]:
    print('WARNING: not all orgnr unique')

# Removing unused objects
del data_loaded


##################################################################
##  Filter to limited liability SMEs
##################################################################
ind = data_brreg['orgform']=='AS'
ind = ind & ((data_brreg['sum_omsetning_EUR'].fillna(0)<=50e6) | (data_brreg['sum_eiendeler_EUR'].fillna(0)<=43e6))
ind = ind & (data_brreg['SUM EIENDELER'].fillna(0)>=500000)

# Excluding industries
ind = ind & (data_brreg['naeringskoder_level_1']!='L') # Real estate activities
ind = ind & (data_brreg['naeringskoder_level_1']!='K') # Financial and insurance activities
ind = ind & (data_brreg['naeringskoder_level_1']!='O') # Public sector
ind = ind & (data_brreg['naeringskoder_level_1']!='D') # Electricity and gas supply
ind = ind & (data_brreg['naeringskoder_level_1']!='E') # Water supply, sewerage, waste
ind = ind & (data_brreg['naeringskoder_level_1']!='MISSING') # Missing
ind = ind & (data_brreg['naeringskoder_level_1']!='0') # companies for investment and holding purposes only

data_brreg = data_brreg[ind].reset_index(drop=True)


##################################################################
##  Preparing data for making variables
##################################################################
# Load data
folder_non_financial = '../data_1_non-financial/'
df_accounting   = pd.read_csv(folder_non_financial+'data_accounting.csv',sep=';',low_memory=False)
df_ceo          = pd.read_csv(folder_non_financial+'data_ceo_complete.csv',sep=';',low_memory=False)
df_chairperson  = pd.read_csv(folder_non_financial+'data_board_chairman.csv',sep=';',low_memory=False)
df_board        = pd.read_csv(folder_non_financial+'data_board.csv',sep=';',low_memory=False)
df_shareholders = pd.read_csv(folder_non_financial+'data_shareholders.csv',sep=';',low_memory=False)

# Changing column names
df_accounting   = df_accounting.rename(columns={'company.org_nr':'orgnr'})
df_ceo          = df_ceo.rename(columns={'company.org_nr':'orgnr'})
df_chairperson  = df_chairperson.rename(columns={'company.org_nr':'orgnr'})
df_board        = df_board.rename(columns={'company.org_nr':'orgnr'})
df_shareholders = df_shareholders.rename(columns={'share_issuer_company.org_nr':'orgnr'})

# Changing column names
df_accounting   = df_accounting.rename(columns={'company_details.business_address_municipality_number':'kommunenr'})
df_ceo          = df_ceo.rename(columns={'person.municipality_code':'kommunenr'})
df_chairperson  = df_chairperson.rename(columns={'person.municipality_code':'kommunenr'})
df_board        = df_board.rename(columns={'person.municipality_code':'kommunenr'})
df_shareholders = df_shareholders.rename(columns={'shareholder_person.municipality_code':'kommunenr'})


# Date 1000-01-01 means that the person has beed in the role from before the data provider
# started collecting data. Changing the value to 1900-01-01 
df_ceo.loc[(df_ceo['person_company_role.from_date']=='1000-01-01'), 'person_company_role.from_date'] = '1900-01-01'
df_chairperson.loc[(df_chairperson['person_company_role.from_date']=='1000-01-01'), 'person_company_role.from_date'] = '1900-01-01'
df_board.loc[(df_board['person_company_role.from_date']=='1000-01-01'), 'person_company_role.from_date'] = '1900-01-01'


# Changing format to datetime
df_ceo['person_company_role.to_date'] =  pd.to_datetime(df_ceo['person_company_role.to_date'])
df_ceo['person_company_role.from_date'] =  pd.to_datetime(df_ceo['person_company_role.from_date'])
df_chairperson['person_company_role.to_date'] =  pd.to_datetime(df_chairperson['person_company_role.to_date'])
df_chairperson['person_company_role.from_date'] =  pd.to_datetime(df_chairperson['person_company_role.from_date'])
df_board['person_company_role.to_date'] =  pd.to_datetime(df_board['person_company_role.to_date'])
df_board['person_company_role.from_date'] =  pd.to_datetime(df_board['person_company_role.from_date'])

# Missing value means that the person still has the role. Setting this to a very high date in the future
df_ceo.loc[pd.isnull(df_ceo['person_company_role.to_date']),'person_company_role.to_date']        = pd.to_datetime('2159-06-16')
df_chairperson.loc[pd.isnull(df_chairperson['person_company_role.to_date']),'person_company_role.to_date']        = pd.to_datetime('2159-06-16')
df_board.loc[pd.isnull(df_board['person_company_role.to_date']),'person_company_role.to_date']  = pd.to_datetime('2159-06-16')


# Changing genders to 'K' for female and 'M' for male
df_ceo.loc[(df_ceo['person.gender_uuid']=='a091eaef-e659-4920-9bdc-d42a75aab765'),'person.gender_uuid'] = 'M'
df_ceo.loc[(df_ceo['person.gender_uuid']=='27ef7f8e-c23d-424b-8050-667a750b1574'),'person.gender_uuid'] = 'K'
df_chairperson.loc[(df_chairperson['person.gender_uuid']=='a091eaef-e659-4920-9bdc-d42a75aab765'),'person.gender_uuid'] = 'M'
df_chairperson.loc[(df_chairperson['person.gender_uuid']=='27ef7f8e-c23d-424b-8050-667a750b1574'),'person.gender_uuid'] = 'K'
df_board.loc[(df_board['person.gender_uuid']=='a091eaef-e659-4920-9bdc-d42a75aab765'),'person.gender_uuid'] = 'M'
df_board.loc[(df_board['person.gender_uuid']=='27ef7f8e-c23d-424b-8050-667a750b1574'),'person.gender_uuid'] = 'K'

# Checking data
if len(df_ceo['person.gender_uuid'].unique())>3:
    print('ERROR in gender for df_ceo')
if len(df_chairperson['person.gender_uuid'].unique())>3:
    print('ERROR in gender for df_chairperson')
if len(df_board['person.gender_uuid'].unique())>3:
    print('ERROR in gender for df_board')

# Checking data
if np.sum(pd.isnull(df_ceo['person_company_role.from_date']))!=0:
    print('ERROR: missing values from date ceo')
if np.sum(pd.isnull(df_chairperson['person_company_role.from_date']))!=0:
    print('ERROR: missing values from date chairperson')
if np.sum(pd.isnull(df_board['person_company_role.from_date']))!=0:
    print('ERROR: missing values from date chairperson')

##################################################################
##  Creating variables
##################################################################
def identify_roles(df, orgnr, balance_date):
    filtered_df = df[df['orgnr'] == orgnr]
    return filtered_df[(filtered_df['person_company_role.from_date'] < balance_date) & 
                       (filtered_df['person_company_role.to_date'] > balance_date)]

def check_kom_fyl(var1, var2):
    if pd.isnull(var1) or pd.isnull(var2):
        return None
    return var1 == var2

def include_non_missing(series):
    non_missing = series.dropna()
    return non_missing.iat[0] if not non_missing.empty else np.nan

def calculate_fylkesnr(x):
    return int(np.floor(np.round(x) / 100)) if pd.notnull(x) else None

# Apply fylkesnr calculation
for df in [df_ceo, df_chairperson, df_accounting, df_board, df_shareholders]:
    df['fylkesnr'] = df['kommunenr'].apply(calculate_fylkesnr)

# Initialize lists
metrics = ['CEO age', 'CEO woman', 'CEO duality', 'CEO on board', 'CEO county', 'two CEOs',
           'chairperson age', 'chairperson woman', 'chairperson county',
           'board size', 'board age avg', 'board age std', 'board women', 'board county', 'board non owners',
           'ownership concentration 1', 'ownership concentration 2', 'ownership CEO', 'ownership chairperson', 'ownership board']

data = {metric: [None] * data_brreg.shape[0] for metric in metrics}

# Reset index 
data_brreg = data_brreg.reset_index(drop=True)

for i in tqdm(range(data_brreg.shape[0])):
    orgnr = data_brreg.at[i,'orgnr']
    accounting_year = data_brreg.at[i,'regnaar']
    balance_date = data_brreg.at[i,'avslutningsdato']
    fylkesnr = df_accounting.loc[(df_accounting['orgnr'] == orgnr) & 
                                 (df_accounting['accounts.accounting_year'] == accounting_year), 'fylkesnr'].iat[0]

    ceo = identify_roles(df_ceo, orgnr, balance_date)
    chairperson = identify_roles(df_chairperson, orgnr, balance_date)
    board = identify_roles(df_board, orgnr, balance_date)
    shareholders = df_shareholders[(df_shareholders['orgnr'] == orgnr) & 
                                   (df_shareholders['company_share_ownership.year'] == accounting_year)]

    if not shareholders.empty:
        data['ownership concentration 1'][i] = shareholders['company_share_ownership.ownership'].mean()
        data['ownership concentration 2'][i] = shareholders['company_share_ownership.ownership'].std(ddof=0)

    if not ceo.empty and not chairperson.empty:
        data['CEO duality'][i] = ceo['person_company_role.person_uuid'].isin(chairperson['person_company_role.person_uuid']).any()

    if not ceo.empty:
        data['two CEOs'][i] = 1 if ceo.shape[0] >= 2 else 0
        data['CEO age'][i] = accounting_year - include_non_missing(ceo['person.birth_year'])
        data['CEO county'][i] = check_kom_fyl(include_non_missing(ceo['fylkesnr']), fylkesnr)
        data['CEO woman'][i] = include_non_missing(ceo['person.gender_uuid']) == 'K'

        if not shareholders.empty:
            data['ownership CEO'][i] = shareholders.loc[shareholders['company_share_ownership.shareholder_entity_uuid'].isin(ceo['person_company_role.person_uuid']), 
                                                        'company_share_ownership.ownership'].sum()

    if not chairperson.empty:
        data['chairperson age'][i] = accounting_year - include_non_missing(chairperson['person.birth_year'])
        data['chairperson county'][i] = check_kom_fyl(include_non_missing(chairperson['fylkesnr']), fylkesnr)
        data['chairperson woman'][i] = include_non_missing(chairperson['person.gender_uuid']) == 'K'

        if not shareholders.empty:
            data['ownership chairperson'][i] = shareholders.loc[shareholders['company_share_ownership.shareholder_entity_uuid'].isin(chairperson['person_company_role.person_uuid']), 
                                                                'company_share_ownership.ownership'].sum()

    if not board.empty:
        data['board size'][i] = board.shape[0]
        data['board county'][i] = (board['fylkesnr'] == fylkesnr).mean() if pd.notnull(fylkesnr) else None
        data['board age avg'][i] = (accounting_year - board['person.birth_year']).mean()
        data['board age std'][i] = (accounting_year - board['person.birth_year']).std(ddof=0)
        data['board women'][i] = (board['person.gender_uuid'] == 'K').mean()

        if not shareholders.empty:
            data['ownership board'][i] = shareholders.loc[shareholders['company_share_ownership.shareholder_entity_uuid'].isin(board['person_company_role.person_uuid']), 
                                                          'company_share_ownership.ownership'].sum()
            data['board non owners'][i] = 1 - board['person_company_role.person_uuid'].isin(shareholders['company_share_ownership.shareholder_entity_uuid']).mean()

        if not ceo.empty:
            data['CEO on board'][i] = ceo['person_company_role.person_uuid'].isin(board['person_company_role.person_uuid']).any()

# Assign data to data_brreg
for metric in metrics:
    data_brreg[metric] = data[metric]

##################################################################
##  Saving data
##################################################################
folder_name = '../data_2_financial_and_non_financial_data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
data_brreg.to_csv(folder_name+'/data_2_financial_and_non_financial_data.csv',index=False,sep=';')



