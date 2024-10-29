# Enhancing credit risk assessments of SMEs with non-financial information
This repository contains all the code necessary for executing our analyses and generating the results detailed in the following paper:

Wahlstrøm, R. R., Becker, L. K., & Fornes, T. N. (2024). Enhancing credit risk assessments of SMEs with non-financial information. Cogent Economics & Finance

## 1_download_non-financial_data.py
The code in this file downloads the non-financial data from Enin AS.

## 2_create_non_financial_variables.py
This file contains code that creates non-financial variables and merges them with the annual financial statements provided by the Norwegian government agency Brønnøysund Register Centre. The data on financial statements are detailed here:

Wahlstrøm, R. R. (2022). Financial statements of companies in Norway. arXiv:2203.12842. https://doi.org/10.48550/arXiv.2203.12842

## 3_create_financial_variables.py
The code in this file creates financial variables and the variables classifying bankruptcy over different horizons (one, two, and three years).

## 4_impute_data.py
The code in this file imputes the missing values of all variables per accounting year using a common approach based on the *k*-nearest neighbor.

## 5_sample_data.py
This file contains the code that implements the synthetic minority oversampling technique (SMOTE) to generate balanced datasets, as described in Section 6.2.1 of the study.

## 6_make_descriptives.py
This code generates the descriptives and the detailed view of the sample presented in the study's Tables 3 and 4, as well as in the Appendix.

## 7_correlation_analysis.py
This file contains the code for generating the correlations in Table 5 of the study.

## 8_analysis_LASSO_LR.R
The code in this file generates the study's main results presented in Section 6.1, using the least absolute shrinkage and selection operator (LASSO) and logistic regression (LR).

## 9_robustness_check_ML.py
This file contains the code for conducting the robustness test using the alternative machine learning methods presented in Section 6.2.2 of the study.

## functions
This folder contains files with functions used by the code in the files mentioned above.

## Data availability statement
The data that support the findings of this study are available from the Norwegian government agency Brønnøysund Register Centre and Enin AS. Restrictions apply to the availability of these data, which were used under license for this study. Data are available from the authors with the permission of the Brønnøysund Register Centre and Enin AS.


<br/><br/>
**Permanent link to the content in this repository:** https://enhancing-credit-risk-2024.ranik.no

