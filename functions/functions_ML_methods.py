import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import os

import time

from sklearn.model_selection import GridSearchCV

from sklearn.inspection import permutation_importance

from sklearn import metrics
from scipy.stats import ks_2samp

import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

def add_metric_values(y_test, pred, new_results_row, index, var_set, k):
    new_results_row.loc[index,(var_set,'AUC')]              = metrics.roc_auc_score(y_test,pred)
    new_results_row.loc[index,(var_set,'AP')]               = metrics.average_precision_score(y_test,pred)
    new_results_row.loc[index,(var_set,'AR')]               = calculate_ar(y_test, pred)
    new_results_row.loc[index,(var_set,'KS statistic')]     = ks_2samp(pred[y_test == 1], pred[y_test == 0]).statistic
    new_results_row.loc[index,(var_set,'Hinge loss')]       = metrics.hinge_loss(np.where(y_test == 1, 1, -1), pred)
    new_results_row.loc[index,(var_set,'Log loss')]         = metrics.log_loss(y_test, pred, normalize=True)
    new_results_row.loc[index,(var_set,'AIC')]              = 2 * k + 2 * metrics.log_loss(y_test, pred, normalize=False)
    new_results_row.loc[index,(var_set,'BIC')]              = k * np.log(len(pred)) + 2 * metrics.log_loss(y_test, pred, normalize=False)
    new_results_row.loc[index,(var_set,'Brier score')]      = metrics.brier_score_loss(y_test, pred)
    new_results_row.loc[index,(var_set,'BSS')]              = 1 - (metrics.brier_score_loss(y_test,pred) / (y_test.mean() * (1 - y_test.mean())))
    decile_rankings = make_decile_rankings(pred,y_test)
    for i in range(decile_rankings.shape[0]):
        new_results_row.loc[index,(var_set,decile_rankings.index[i])] = decile_rankings.iloc[i]
    return new_results_row

def calculate_ar(y_true, y_pred):
    # Sort by predicted probabilities
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    data = data.sort_values(by='y_pred', ascending=False)
    
    # Calculate cumulative true positives and total positives
    data['cum_true_positives'] = data['y_true'].cumsum()
    total_positives = data['y_true'].sum()
    
    # Calculate cumulative percentage of positives and population
    data['cum_percentage_positives'] = data['cum_true_positives'] / total_positives
    data['cum_percentage_population'] = np.arange(1, len(data) + 1) / len(data)
    
    # Calculate areas
    area_model = np.trapz(data['cum_percentage_positives'], data['cum_percentage_population'])
    area_random = 0.5
    area_perfect = 1.0
    
    # Calculate Accuracy Ratio (AR)
    ar = (area_model - area_random) / (area_perfect - area_random)
    return ar

def make_decile_rankings(pred,y_test):
    num_deciles = 10
    num_last_deciles_to_group = 5
    temp = pd.DataFrame({'targets': y_test, 'predicted': pred})
    temp = temp.sort_values('predicted', ascending=False).reset_index(drop=True)

    num_labels = len(pd.qcut(temp['predicted'], num_deciles, duplicates='drop').unique())
    labels=list(range(1, num_labels+1))[::-1]

    temp['decile'] = pd.qcut(temp['predicted'], num_deciles, duplicates='drop', labels=labels)
    decile_rankings = pd.Series()
    for i in range(1, (num_deciles-5)+1):
        decile_rankings['Decile '+str(i)] = temp.loc[temp['decile']==i,'targets'].sum()/temp['targets'].sum()
    deciles_last = list(range(num_deciles-num_last_deciles_to_group+1,num_deciles+1))
    decile_rankings['Decile {}-{}'.format(np.min(deciles_last),np.max(deciles_last))] = temp.loc[temp['decile'].isin(deciles_last),'targets'].sum()/temp['targets'].sum()
    return decile_rankings

def get_tuned_ML_hyperparameters(parameters_for_functions):

    estimation_method       = parameters_for_functions['estimation_method']
    financial_variable_set  = parameters_for_functions['financial_variable_set']
    prediction_horizon      = parameters_for_functions['prediction_horizon']
    year                    = parameters_for_functions['year']
    X_train                 = parameters_for_functions['X_train']
    y_train                 = parameters_for_functions['y_train']

    file_name_tuned_parameters = estimation_method+'_'+financial_variable_set+'_'+prediction_horizon+'_'+str(year)+'.csv'
                
    folder_name_tuned_parameters = 'tuned_hyperparameters'
    if not os.path.exists(folder_name_tuned_parameters):
        os.makedirs(folder_name_tuned_parameters)

    files = os.listdir(folder_name_tuned_parameters)

    if file_name_tuned_parameters not in files:
        time_year = time.time()
        print('Tuning hyperparameters for {}'.format(file_name_tuned_parameters))

        if estimation_method == 'XGBoost':
            param_grid = {
                'learning_rate': (0.01,0.1,0.3), # step size shrinkage parameter
                'subsample':  (0.7,1),
                'gamma':(0.0,0.2,0.4),
                'reg_lambda':(1,2,3),
                'n_estimators': (50,100),
                'max_depth': (3,6,8), # Maximum depth of the decision trees
            }
            model = xgboost.XGBClassifier(
                    objective= 'binary:logistic',
                    eval_metric='logloss',
                    random_state=1,
                    )

        elif estimation_method == 'AdaBoost':
            param_grid = {
                'n_estimators': (50, 75),
                'learning_rate': (0.1, 1.0),
            }
            model = AdaBoostClassifier(random_state=1)

        elif estimation_method == 'RandomForest':
            param_grid = {
                'n_estimators': (50, 75),  # Number of trees in the forest
                'min_samples_split': (2, 5),  # Minimum number of samples required to split an internal node
                'min_samples_leaf': (1, 2),  # Minimum number of samples required to be at a leaf node
            }
            model = RandomForestClassifier(random_state=1)

        elif estimation_method == 'BaggingClassifier':
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 1.0],
                'max_features': [0.5, 1.0]
            }
            model = BaggingClassifier(random_state=1)

        # Performing grid search
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring='roc_auc',
            refit=True,
            cv=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Saving tuned hyperparameters
        tuned_hyperparameters = pd.Series(grid_search.best_params_)
        tuned_hyperparameters.to_csv(folder_name_tuned_parameters + '/' + file_name_tuned_parameters, sep=';')

        # Printing elapset time
        print('{}: {} minutes'.format(file_name_tuned_parameters,np.round((time.time() - time_year)/60,2)))
    
    else:
        tuned_hyperparameters = pd.read_csv(folder_name_tuned_parameters+'/'+file_name_tuned_parameters,sep=';',index_col=0)
        tuned_hyperparameters = tuned_hyperparameters['0']

    return tuned_hyperparameters



def train_models_and_get_feature_importance(parameters_for_functions):

    estimation_method       = parameters_for_functions['estimation_method']
    X_train                 = parameters_for_functions['X_train']
    y_train                 = parameters_for_functions['y_train']
    tuned_hyperparameters   = parameters_for_functions['tuned_hyperparameters']
    X_train_fin             = parameters_for_functions['X_train_fin']

    financial_variable_set  = parameters_for_functions['financial_variable_set']
    prediction_horizon      = parameters_for_functions['prediction_horizon']
    year                    = parameters_for_functions['year']

    temp_name_for_print = estimation_method+'_'+financial_variable_set+'_'+prediction_horizon+'_'+str(year)

    time_year = time.time()
    print('Training model {}'.format(temp_name_for_print))

    if estimation_method == 'XGBoost':
        params = {
            'learning_rate': tuned_hyperparameters.loc['learning_rate'],
            'subsample': tuned_hyperparameters.loc['subsample'],
            'gamma': tuned_hyperparameters.loc['gamma'],
            'reg_lambda': tuned_hyperparameters.loc['reg_lambda'],
            'n_estimators': int(tuned_hyperparameters.loc['n_estimators']),
            'max_depth': int(tuned_hyperparameters.loc['max_depth']),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 1
        }
        model = xgboost.XGBClassifier(**params).fit(X_train, y_train)
        model_fin = xgboost.XGBClassifier(**params).fit(X_train_fin, y_train)
        feature_importances = model.feature_importances_


    elif estimation_method == 'AdaBoost':
        params = {
            'n_estimators': int(tuned_hyperparameters.loc['n_estimators']),
            'learning_rate': float(tuned_hyperparameters.loc['learning_rate']),
            'random_state': 1
        }
        model = AdaBoostClassifier(**params).fit(X_train, y_train)
        model_fin = AdaBoostClassifier(**params).fit(X_train_fin, y_train)
        feature_importances = model.feature_importances_

    elif estimation_method == 'RandomForest':
        params = {
            'n_estimators': int(tuned_hyperparameters.loc['n_estimators']),
            'min_samples_split': int(tuned_hyperparameters.loc['min_samples_split']),
            'min_samples_leaf': int(tuned_hyperparameters.loc['min_samples_leaf']),
            'random_state': 1,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params).fit(X_train, y_train)
        model_fin = RandomForestClassifier(**params).fit(X_train_fin, y_train)
        feature_importances = model.feature_importances_

    elif estimation_method == 'BaggingClassifier':
        params = {
            'n_estimators': int(tuned_hyperparameters.loc['n_estimators']),
            'max_samples': float(tuned_hyperparameters.loc['max_samples']),
            'max_features': float(tuned_hyperparameters.loc['max_features']),
            'random_state': 1,
            'n_jobs': -1
        }
        model = BaggingClassifier(**params).fit(X_train, y_train)
        model_fin = BaggingClassifier(**params).fit(X_train_fin, y_train)
        feature_importances = permutation_importance(model, X_train, y_train, random_state=1).importances_mean

    # Printing elapset time
    print('{}: {} minutes'.format(temp_name_for_print,np.round((time.time() - time_year)/60,2)))

    return model,model_fin,feature_importances