import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import requests
import json

import os

from io import StringIO

# Autorisasjon API (Enin AS)
def print_obj(obj):
    print(json.dumps(obj, indent=4, ensure_ascii=False))

# Printing system status to see if the server is running
system_status = requests.get("https://api.enin.ai/datasets/v1/system-status").json()
print_obj(system_status)

# Authenticating
# For not sharing my key on GitHub, I have saved my key in an 
# unavailable file
client_data = pd.read_csv('../enin_keys.csv',sep=';',index_col=None)
auth = (client_data['client_id'].iloc[0], client_data['client_secret'].iloc[0])

# Printing status to let you know if you are authenticated
auth_status = requests.get(
    "https://api.enin.ai/datasets/v1/auth-status",
    auth=auth
).json()
print_obj(auth_status)

# Define function
def convert_to_pd_and_remove_duplicates(data):
    # Convert to pandas data frame
    StringData = StringIO(data)
    data = pd.read_csv(StringData, sep =",")
    # If any are registered more than once, keep only one version of each 
    return data.drop_duplicates().reset_index(drop=True)


# Defining years for downloading data
years_for_downloading = "IN:"
from_year   = 2014
to_year     = 2022
for year in range(from_year,to_year+1):
    years_for_downloading = years_for_downloading+str(year)+","
years_for_downloading = years_for_downloading[:-1]


##########################################
## Make folder for saving data
##########################################
folder_name = '../data_1_non-financial'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

######################################
## Accounting information
######################################
data_accounting = requests.get(
"https://api.enin.ai/datasets/v1/dataset/accounts-composite",
params={
"response_file_type": "csv",
"accounts.accounting_year": years_for_downloading,
"keep_only_fields": ','.join(
[
                "company.org_nr",
                "accounts.accounting_year",
                "company_details.employees",
                "accounts.accounting_to_date",
                "accounts.accounting_from_date",
                "company_details.business_address_municipality_number",
                "accounts_balance_sheet.total_assets",
                "accounts_income_statement.total_operating_income",
                "organization_type.organization_type_code",
]
)
},
auth=auth,
).content.decode()
data_accounting = convert_to_pd_and_remove_duplicates(data_accounting)

# Saving to .csv
data_accounting.to_csv(folder_name+'/data_accounting.csv',sep=';',index=False)

# Deleting to free memory
del data_accounting


######################################
## Shareholders
######################################
data_shareholders = requests.get(
"https://api.enin.ai/datasets/v1/dataset/company-share-ownership-composite",
    params={
        "company_share_ownership.year": years_for_downloading,
        "response_file_type": "csv",
        "keep_only_fields": ','.join(
            [
                "shareholder_company.org_nr",
                "share_issuer_company.org_nr",
                "company_share_ownership.year",
                "company_share_ownership.share_count", 
                "company_share_ownership.total_share_count",
                "company_share_ownership.ownership",
                "company_share_ownership.shareholder_entity_uuid",
                "shareholder_person.municipality_code",
                "shareholder_person.postal_code",
             ]
    )
    },
auth=auth,
).content.decode()
data_shareholders = convert_to_pd_and_remove_duplicates(data_shareholders)

# Saving to .csv
data_shareholders.to_csv(folder_name+'/data_shareholders.csv',sep=';',index=False)

# Deleting to free memory
del data_shareholders


######################################
## CEO
######################################
# Norwegian: "Daglig leder" 
data_ceo = requests.get(
    "https://api.enin.ai/datasets/v1/dataset/person-company-role-composite",
    params={
        "company_role.company_role_key": "EQ:ceo",
        "response_file_type": "csv",
        "keep_only_fields": ','.join(
            [
                "person.municipality_code",
                "person.postal_code",
                "person.birth_year",
                "company.org_nr",
                "company_role.company_role_key",
                "person.gender_uuid", 
                'person_company_role.from_date',
                'person_company_role.to_date', 
                'person_company_role.person_uuid',      
             ]
    )
},
auth=auth,
).content.decode()
data_ceo = convert_to_pd_and_remove_duplicates(data_ceo)

# Norwegian: "Forretningsfører"
data_business_manager = requests.get(
    "https://api.enin.ai/datasets/v1/dataset/person-company-role-composite",
    params={
        "company_role.company_role_key": "EQ:business_manager",
        "response_file_type": "csv",
        "keep_only_fields": ','.join(
            [
                "person.municipality_code",
                "person.postal_code",
                "person.birth_year",
                "company.org_nr",
                "company_role.company_role_key",
                "person.gender_uuid", 
                'person_company_role.from_date',
                'person_company_role.to_date', 
                'person_company_role.person_uuid',      
             ]
    )
},
auth=auth,
).content.decode()
data_business_manager = convert_to_pd_and_remove_duplicates(data_business_manager)

# Norwegian: "Kontaktperson"
data_contact_person = requests.get(
    "https://api.enin.ai/datasets/v1/dataset/person-company-role-composite",
    params={
        "company_role.company_role_key": "EQ:contact_person",
        "response_file_type": "csv",
        "keep_only_fields": ','.join(
            [
                "person.municipality_code",
                "person.postal_code",
                "person.birth_year",
                "company.org_nr",
                "company_role.company_role_key",
                "person.gender_uuid", 
                'person_company_role.from_date',
                'person_company_role.to_date', 
                'person_company_role.person_uuid',      
             ]
    )
},
auth=auth,
).content.decode()
data_contact_person = convert_to_pd_and_remove_duplicates(data_contact_person)

# Merging to one data frame with ceo data
data_ceo_complete = pd.concat([data_ceo,data_business_manager],axis=0).reset_index(drop=True)
data_ceo_complete = pd.concat([data_ceo_complete,data_contact_person],axis=0).reset_index(drop=True)
if data_ceo_complete.shape[0]!= (data_ceo.shape[0]+data_business_manager.shape[0]+data_contact_person.shape[0]):
    print('ERROR merging ceo data')

# Remove any duplicates
data_ceo_complete = data_ceo_complete.drop_duplicates(list(data_ceo_complete)[:-1],keep='first').reset_index(drop=True)

# Saving to .csv
data_ceo_complete.to_csv(folder_name+'/data_ceo_complete.csv',sep=';',index=False)

# Deleting to free memory
del data_ceo
del data_business_manager
del data_contact_person
del data_ceo_complete

######################################
## Board
######################################
data_board_chairman = requests.get( 
    "https://api.enin.ai/datasets/v1/dataset/person-company-role-composite",
    params={
        "company_role.company_role_key": "EQ:board_chairman",
        "response_file_type": "csv",
        "keep_only_fields": ','.join(
            [
                "person.municipality_code",
                "person.postal_code",
                "person.birth_year",
                "company.org_nr",
                "company_role.company_role_key",
                "person.gender_uuid", 
                'person_company_role.from_date',
                'person_company_role.to_date', 
                'person_company_role.person_uuid',        
             ]
    )
},
auth=auth,
).content.decode()
data_board_chairman = convert_to_pd_and_remove_duplicates(data_board_chairman)

data_deputy_chairman = requests.get(
    "https://api.enin.ai/datasets/v1/dataset/person-company-role-composite",
    params={
        "company_role.company_role_key": "EQ:deputy_chairman",
        "response_file_type": "csv",
        "keep_only_fields": ','.join(
            [
                "person.municipality_code",
                "person.postal_code",
                "person.birth_year",
                "company.org_nr",
                "company_role.company_role_key",
                "person.gender_uuid", 
                'person_company_role.from_date',
                'person_company_role.to_date', 
                'person_company_role.person_uuid',       
             ]
    )
},
auth=auth,
).content.decode()
data_deputy_chairman = convert_to_pd_and_remove_duplicates(data_deputy_chairman)

data_board_member = requests.get(
    "https://api.enin.ai/datasets/v1/dataset/person-company-role-composite",
    params={
        "company_role.company_role_key": "EQ:board_member",
        "response_file_type": "csv",      
        "keep_only_fields": ','.join(
            [
                "person.municipality_code",
                "person.postal_code",
                "person.birth_year",
                "company.org_nr",
                "company_role.company_role_key",
                "person.gender_uuid", 
                'person_company_role.from_date',
                'person_company_role.to_date', 
                'person_company_role.person_uuid',       
             ]
    )
},
auth=auth,
).content.decode()
data_board_member = convert_to_pd_and_remove_duplicates(data_board_member)

# Merging to one data frame with board data
data_board  = pd.concat([data_board_member,data_deputy_chairman],axis=0).reset_index(drop=True)
data_board  = pd.concat([data_board,data_board_chairman],axis=0).reset_index(drop=True)
if data_board.shape[0]!= (data_board_member.shape[0]+data_deputy_chairman.shape[0]+data_board_chairman.shape[0]):
    print('ERROR merging board data')

# Remove any duplicates
data_board = data_board.drop_duplicates(list(data_board)[:-1],keep='first').reset_index(drop=True)

# Saving to .csv
data_board_chairman.to_csv(folder_name+'/data_board_chairman.csv',sep=';',index=False)
data_board.to_csv(folder_name+'/data_board.csv',sep=';',index=False)

# Deleting to free memory
del data_board_member
del data_board_chairman
del data_deputy_chairman
del data_board



