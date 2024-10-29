import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

import os

import matplotlib.pyplot as plt

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *
from functions_ML_methods import *

# Set font to Times New Roman
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

# Set global font size for plots
plt.rcParams.update({'font.size':26})

############################################################
## Load data
############################################################
data = pd.read_csv('../data_4_imputed/data_imputed.csv',sep=';',low_memory=False)

# Make folders for saving results
folder_name = 'results/ML'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Make folders for saving prediction performance results
folder_name_perf = 'results/ML/model_performance'
if not os.path.exists(folder_name_perf):
    os.makedirs(folder_name_perf)

measure_names = [
    'AUC',
    'AP',
    'AR',
    'KS statistic',
    'Hinge loss',
    'Log loss',
    'AIC',
    'BIC',
    'Brier score',
    'BSS',
    'Decile 1',
    'Decile 2',
    'Decile 3',
    'Decile 4',
    'Decile 5',
    'Decile 6-10',
]

financial_variable_sets = [
    'altman_1968',
    'altman_and_sabato_2007',
    'paraschiv_2021',
]

prediction_horizons = [
    'bankrupt_1',
    'bankrupt_2',
    'bankrupt_3',
]

estimation_methods = [
    'XGBoost',
    'AdaBoost',
    'BaggingClassifier',
    'RandomForest',
]

for estimation_method in estimation_methods:
    print('Estimation method {}'.format(estimation_method))

    for prediction_horizon in tqdm(prediction_horizons):

        columns = []
        columns.append((None, 'Benchmark set'))
        columns.append((None, 'Year'))
        for temp in ['Financial and non-financial variables','Exclusively financial variables']:
            for measure_name in measure_names:
                columns.append((temp, measure_name))
        columns = pd.MultiIndex.from_tuples(columns)
        results_df = pd.DataFrame(columns=columns)


        for financial_variable_set in financial_variable_sets:

            if financial_variable_set == 'altman_1968':
                variables_financial = get_variables_altman_1968()
            elif financial_variable_set == 'altman_and_sabato_2007':
                variables_financial = get_variables_altman_and_sabato_2007()
            elif financial_variable_set == 'paraschiv_2021':
                variables_financial = get_variables_paraschiv_2021()
            else:
                print('ERROR in defining financial_variable_set')
            variables_with_non_financials = get_non_financial_variables() + variables_financial

            for year in range(2018,2019+1):
                
                # Defining test data
                data_test = data[data['regnaar']==year].reset_index(drop=True)
                X_test      = data_test[variables_with_non_financials].astype(float)
                X_test_fin  = data_test[variables_financial].astype(float)
                y_test      = data_test[prediction_horizon].astype(float)

                # Defining training data
                data_train = data[(data['regnaar']<year)&(data['regnaar']>=(year-4))].reset_index(drop=True)
                X_train     = data_train[variables_with_non_financials].astype(float)
                X_train_fin = data_train[variables_financial].astype(float)
                y_train     = data_train[prediction_horizon].astype(float)

                parameters_for_functions = {}
                parameters_for_functions['estimation_method']       = estimation_method
                parameters_for_functions['financial_variable_set']  = financial_variable_set
                parameters_for_functions['prediction_horizon']      = prediction_horizon
                parameters_for_functions['year']                    = year
                parameters_for_functions['X_train']                 = X_train
                parameters_for_functions['X_train_fin']             = X_train_fin
                parameters_for_functions['y_train']                 = y_train

                # Tune or get hyperparameters
                parameters_for_functions['tuned_hyperparameters'] = get_tuned_ML_hyperparameters(parameters_for_functions)

                # Train models and get feature importance
                model, model_fin, feature_importances = train_models_and_get_feature_importance(parameters_for_functions)

                feature_names = X_train.columns

                # Sort the feature importances in descending order
                indices = np.argsort(feature_importances)[::-1]

                # Get the top 10 features
                top_n = 10
                top_indices = indices[:top_n]
                top_features = [feature_names[i] for i in top_indices]
                top_importances = feature_importances[top_indices]

                # Create a horizontal bar plot
                plt.figure(figsize=(10, 6))
                bars = plt.barh(range(top_n), top_importances, align='center')

                # Highlight non-financial variables
                highlight_features = get_non_financial_variables() 
                for bar, feature in zip(bars, top_features):
                    if feature in highlight_features:
                        bar.set_color('purple')
                    else:
                        bar.set_color('green')

                # Remove the border
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                plt.yticks(range(top_n), top_features)
                plt.xlabel('Importance Score', x=1.0, horizontalalignment='right', labelpad=10, bbox=dict(facecolor='white', edgecolor='none', pad=5))
                plt.tight_layout()
                plt.savefig(folder_name+'/'+estimation_method+'_'+financial_variable_set+'_'+prediction_horizon+'_'+str(year)+'.png')
                plt.close() # So the figure does not show in kernel

                # Add performance metrics
                index = len(results_df)
                new_results_row = pd.DataFrame(columns=results_df.columns)

                if financial_variable_set == 'altman_1968':
                    new_results_row.loc[index,(None,'Benchmark set')] = 'Altman (1968)'
                elif financial_variable_set == 'altman_and_sabato_2007':
                    new_results_row.loc[index,(None,'Benchmark set')] = 'Altman and Sabato (2007)'
                elif financial_variable_set == 'paraschiv_2021':
                    new_results_row.loc[index,(None,'Benchmark set')] = 'Paraschiv et al. (2023)'

                new_results_row.loc[index,(None,'Year')]                = year

                var_set = 'Financial and non-financial variables'
                pred    = model.predict_proba(X_test)[:,1]
                k = X_test.shape[1] + 1 # Number of parameters (including intercept)
                new_results_row = add_metric_values(y_test, pred, new_results_row, index, var_set, k)

                var_set = 'Exclusively financial variables'
                pred    = model_fin.predict_proba(X_test_fin)[:,1]
                k = X_test_fin.shape[1] + 1 # Number of parameters (including intercept)
                new_results_row = add_metric_values(y_test, pred, new_results_row, index, var_set, k)

                results_df = pd.concat([results_df,new_results_row],axis=0)

        results_df.T.to_excel(folder_name_perf+'/'+estimation_method+'_'+prediction_horizon+'.xlsx',header=False)


