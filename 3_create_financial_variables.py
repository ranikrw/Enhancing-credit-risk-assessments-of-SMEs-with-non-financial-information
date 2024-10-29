import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

import os

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *

####################################################
## Load data
####################################################
data = pd.read_csv('../data_2_financial_and_non_financial_data/data_2_financial_and_non_financial_data.csv', sep=';', low_memory=False)

data = data[(data['regnaar']>=2014)&(data['regnaar']<=2019)].reset_index(drop=True)

#################################
## Defining bankruptcy variables
#################################
data['konkursdato'] = pd.to_datetime(data['konkursdato'], format='%Y-%m-%d %H:%M:%S')
data['avslutningsdato'] = pd.to_datetime(data['avslutningsdato'], format='%Y-%m-%d')

data['diff_days'] = (data['konkursdato'] - data['avslutningsdato']).dt.days
data['diff_years'] = data['diff_days'] / 365.25  # Using 365.25 to account for leap years

data['bankrupt_1'] = (data['diff_years']<=1).astype(int)
data['bankrupt_2'] = ((data['diff_years']>1)&(data['diff_years']<=2)).astype(int)
data['bankrupt_3'] = ((data['diff_years']>2)&(data['diff_years']<=3)).astype(int)

##############################
## Defining accounting values
##############################
# For the accounting data, missing values means that it is zero. 
# We therefore use .fillna(0) at the end for the accounting variables

# Usually 'Bankinnskudd, kontanter og lignende' catches everything, but 
# sometimes 'Sum bankinnskudd, kontanter og lignende' needs to be used instead
string1 = 'Bankinnskudd, kontanter og lignende'
string2 = 'Sum bankinnskudd, kontanter og lignende'
bankinnskudd_kontanter_og_lignende = pd.Series([None]*data.shape[0])
for i in tqdm(range(data.shape[0])):
    if pd.isnull(data[string1].iloc[i])==False:
        bankinnskudd_kontanter_og_lignende[i] = data[string1].iloc[i]
    elif pd.isnull(data[string2].iloc[i])==False:
        bankinnskudd_kontanter_og_lignende[i] = data[string2].iloc[i]
    else: # if both is 'None'
        bankinnskudd_kontanter_og_lignende[i] = np.double(0)

skyldige_offentlige_avgifter            = data['Skyldige offentlige avgifter'].fillna(0)
leverandorgjeld                         = data['Leverandoergjeld'].fillna(0)
sum_kortsiktig_gjeld                    = data['Sum kortsiktig gjeld'].fillna(0)
sum_inntekter                           = data['Sum inntekter'].fillna(0)
sum_innskutt_egenkapital                = data['Sum innskutt egenkapital'].fillna(0)
sum_egenkapital                         = data['Sum egenkapital'].fillna(0)
sum_eiendeler                           = data['SUM EIENDELER'].fillna(0)
avskrivninger                           = data['Avskrivning paa varige driftsmidler og immaterielle eiendeler'].fillna(0)
ordinaert_resultat_foer_skattekostnad   = data['Ordinaert resultat foer skattekostnad'].fillna(0)
ordinaert_resultat_etter_skattekostnad  = data['Ordinaert resultat etter skattekostnad'].fillna(0)
sum_gjeld                               = data['Sum gjeld'].fillna(0)
arsresultat                             = data['Aarsresultat'].fillna(0)
annen_rentekostnad                      = data['Sum finanskostnader'].fillna(0)
annen_renteinntekt                      = data['Annen renteinntekt'].fillna(0)
utbytte                                 = data['Utbytte'].fillna(0)
opptjent_egenkapital                    = data['Sum opptjent egenkapital'].fillna(0)
gjeld_til_kredittinstitusjoner          = data['Gjeld til kredittinstitusjoner'].fillna(0)
salgsinntekt                            = data['Salgsinntekt'].fillna(0)
lonnskostnad                            = data['Loennskostnad'].fillna(0)
sum_varer                               = data['Sum varer'].fillna(0)
kundefordringer                         = data['Kundefordringer'].fillna(0)
sum_omlopsmidler                        = data['Sum omloepsmidler'].fillna(0)
nedskrivninger                          = data['Nedskrivning av varige driftsmidler og immaterielle eiendeler'].fillna(0)

EBIT = ordinaert_resultat_foer_skattekostnad + annen_rentekostnad - annen_renteinntekt
EBITDA = EBIT + avskrivninger + nedskrivninger

####################################################
## Define function for creating ratios
####################################################
def make_ratio(numerator,denumerator):
    ratio = [None]*len(numerator)
    for i in range(len(numerator)):
        if denumerator[i]!=0:
            ratio[i] = numerator[i] / denumerator[i]
        else: #denumerator[i]==0:
            if numerator[i]>0:
                ratio[i] = np.inf
            elif numerator[i]<0:
                ratio[i] = -np.inf
            else: # numerator[i]==0:
                ratio[i] = 0
    return pd.Series(ratio)

############################################################
## Creating variables of Paraschiv et al. (2021)
############################################################
data['accounts payable / total assets'] = make_ratio(leverandorgjeld,sum_eiendeler)
data['dummy; one if total liability exceeds total assets'] = (sum_gjeld > sum_eiendeler).astype(int)

numerator = sum_kortsiktig_gjeld-bankinnskudd_kontanter_og_lignende
data['(current liabilities - short-term liquidity) / total assets'] = make_ratio(numerator,sum_eiendeler)

data['net income / total assets'] = make_ratio(arsresultat,sum_eiendeler)
data['public taxes payable / total assets'] = make_ratio(skyldige_offentlige_avgifter,sum_eiendeler)
data['interest expenses / total assets'] = make_ratio(annen_rentekostnad,sum_eiendeler)
data['dummy; one if paid-in equity is less than total equity'] = (sum_innskutt_egenkapital < sum_egenkapital).astype(int)

temp = data['age_in_days'].copy()
ind = temp<0
if np.sum(ind)!=0:
    print('For {} observations, age is negative. Setting these to age zero.'.format(np.sum(ind)))
    temp[ind] = 0
data['log(age in years)'] = np.log((temp+1)/365)

data['inventory / current assets'] = make_ratio(sum_varer,sum_omlopsmidler)
data['short-term liquidity / current assets'] = make_ratio(bankinnskudd_kontanter_og_lignende,sum_omlopsmidler)

############################################################
## Creating variables of Altman and Sabato (2007)
############################################################
data['current liabilities / total equity'] = make_ratio(sum_kortsiktig_gjeld,sum_egenkapital)
data['EBITDA / interest expense'] = make_ratio(EBITDA,annen_rentekostnad)
data['EBITDA / total assets'] = make_ratio(EBITDA,sum_eiendeler)
data['retained earnings / total assets'] = make_ratio(opptjent_egenkapital,sum_eiendeler)
data['short-term liquidity / total assets'] = make_ratio(bankinnskudd_kontanter_og_lignende,sum_eiendeler)

############################################################
## Creating variables of Altman (1968)
############################################################
data['EBIT / total assets'] = make_ratio(EBIT,sum_eiendeler)

# This variable is not created here as it is created above for 
# the Altman and Sabato (2007) model
# data['retained earnings / total assets'] = make_ratio(opptjent_egenkapital,sum_eiendeler)

data['sales / total assets'] = make_ratio(salgsinntekt,sum_eiendeler)
data['total equity / total liabilities'] = make_ratio(sum_egenkapital,sum_gjeld)

numerator = sum_omlopsmidler - sum_kortsiktig_gjeld
data['working capital / total assets'] = make_ratio(numerator,sum_eiendeler)

############################################################
## Winzorise ratio variables
############################################################
interval_winsorizing_ratios = [0.01,0.99] # In numbers, so 0.01 = restricting at 1%

# Defining variables that shall be winsorized
ratio_variables_to_winsorize = get_variables_paraschiv_2021()
ratio_variables_to_winsorize = ratio_variables_to_winsorize + get_variables_altman_1968()
ratio_variables_to_winsorize = ratio_variables_to_winsorize + get_variables_altman_and_sabato_2007()
ratio_variables_to_winsorize = list(np.unique(ratio_variables_to_winsorize)) # Making sure all are unique

# Removing variables that are not ratios, as only
# ratios shall be winsorized
ratio_variables_to_winsorize.remove('dummy; one if total liability exceeds total assets')
ratio_variables_to_winsorize.remove('dummy; one if paid-in equity is less than total equity')
ratio_variables_to_winsorize.remove('log(age in years)')

# Winsorizing, per accounting year
# (Before winsorizing, inf and -inf to maximum and minimum, respectively, values)
data_winsorized = pd.DataFrame(columns=data.columns)
for regnaar in tqdm(data['regnaar'].unique()):
    data_regnaar = data[data['regnaar']==regnaar].copy()
    for var in ratio_variables_to_winsorize:
        ratio = data_regnaar[var]
        
        # Setting inf and -inf to maximum and minimum, respectively, values
        ratio = ratio.replace(np.inf,np.max(ratio[ratio != np.inf]))
        ratio = ratio.replace(-np.inf,np.max(ratio[ratio != -np.inf]))

        lower = ratio.quantile(interval_winsorizing_ratios[0])
        upper = ratio.quantile(interval_winsorizing_ratios[1])

        data_regnaar[var] = ratio.clip(lower=lower, upper=upper)
    data_winsorized = pd.concat([data_winsorized,data_regnaar],axis=0)

# Controlling data
if data.shape[0]!=data_winsorized.shape[0]:
    print('ERROR: not same num rows after winsorizing')
if (data.shape[0])!=data_winsorized.shape[0]:
    print('ERROR: not right num cols after winsorizing')
if np.sum(np.sum(pd.isnull(data_winsorized[ratio_variables_to_winsorize])))!=0:
    print('ERROR: some ratio values are still missing/NULL')

# Checking
if data_winsorized.shape[0]!=data.shape[0]:
    print('ERROR when winsorizing')
if data_winsorized.shape[1]!=data.shape[1]:
    print('ERROR when winsorizing')
if np.sum(data_winsorized['orgnr'])!=np.sum(data['orgnr']):
    print('ERROR when winsorizing')
if np.sum(data_winsorized['regnaar'])!=np.sum(data['regnaar']):
    print('ERROR when winsorizing')
if np.sum(data_winsorized['log(age in years)'])!=np.sum(data['log(age in years)']):
    print('ERROR when winsorizing')

data = data_winsorized.copy()
del data_winsorized


############################################################
## Taking natural logarithm
############################################################
variables_to_log = [
    'CEO age',
    'chairperson age',
    'board size',
    'board age avg',
    # 'number of shareholders',
]

for var in variables_to_log:
    data[var] = np.log(data[var])


############################################################
## Convert bool to int
############################################################
def fun_convert_dummy_to_1_and_0(x):
    if pd.isnull(x):
        return None
    else:
        return int(x)

variables_to_convert = [
    'CEO duality',
    'CEO on board',
    'CEO woman',
    'CEO county',
    'chairperson woman',
    'chairperson county',
]

for var in variables_to_convert:
    temp = np.sum(data[var])
    data[var] = data[var].apply(fun_convert_dummy_to_1_and_0)
    if temp != np.sum(data[var]):
        print('ERROR in converting {} from TRUE/FALSE to 1/0'.format(var))

############################################################
## Save to file
############################################################
folder_name = '../data_3_for_analysis'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

data.to_csv(folder_name+'/data_with_missing.csv',index=False,sep=';')

