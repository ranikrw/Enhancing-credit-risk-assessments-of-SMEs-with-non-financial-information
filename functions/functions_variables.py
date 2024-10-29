import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team


def get_non_financial_variables_management_structure():
    variables_non_financial = [
        'CEO age',
        'CEO woman',
        'CEO duality',
        'CEO on board',
        'CEO county',
        'two CEOs',
        ]
    return variables_non_financial

def get_non_financial_variables_board_structure():
    variables_non_financial = [
        'chairperson age',
        'chairperson woman',
        'chairperson county',
        'board size',
        'board age avg',
        'board age std',
        'board women',
        'board county',
        'board non-owners',
        ]
    return variables_non_financial

def get_non_financial_variables_ownership_structure():
    variables_non_financial = [
        'ownership concentration 1',
        'ownership concentration 2',
        'ownership CEO',
        'ownership chairperson',
        'ownership board',
        ]
    return variables_non_financial

def get_non_financial_variables():
    variables_non_financial = get_non_financial_variables_management_structure()
    variables_non_financial = variables_non_financial + get_non_financial_variables_board_structure()
    variables_non_financial = variables_non_financial + get_non_financial_variables_ownership_structure()
    return variables_non_financial

def get_variables_paraschiv_2021():
    variables = [
        '(current liabilities - short-term liquidity) / total assets',
        'accounts payable / total assets',
        'dummy; one if paid-in equity is less than total equity',
        'dummy; one if total liability exceeds total assets',
        'interest expenses / total assets',
        'inventory / current assets',
        'log(age in years)',
        'net income / total assets',
        'public taxes payable / total assets',
        'short-term liquidity / current assets',
    ]
    return variables

def get_variables_altman_and_sabato_2007():
    variables = [
        'current liabilities / total equity',
        'EBITDA / interest expense',
        'EBITDA / total assets',
        'retained earnings / total assets',
        'short-term liquidity / total assets',
    ]
    return variables

def get_variables_altman_1968():
    variables = [
        'EBIT / total assets',
        'retained earnings / total assets',
        'sales / total assets',
        'total equity / total liabilities',
        'working capital / total assets',
    ]
    return variables

