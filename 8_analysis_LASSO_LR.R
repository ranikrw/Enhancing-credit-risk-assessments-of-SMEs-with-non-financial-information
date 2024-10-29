library(glmnet)
library(pROC)
library(DescTools)
library(car)
library(precrec)
library(Metrics)
library(selectiveInference)


Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre-1.8') # for 64-bit version
library(xlsx)

library(MLmetrics) # For PRAUC

library(dplyr)

library(pscl) # McFaddens R squared

library(stringr) # for function "str_count"

library(writexl) # Write to Excel

library(ggplot2) # For plotting
library(reshape) # Reshaping data frame for plotting with ggplot

# Including functions
source("functions/functions_analysis.R")

# Get variable sets
variables_non_financial           = get_variables_non_financial()
variables_altman_1968             = get_variables_altman_1968()
variables_altman_and_sabato_2007  = get_variables_altman_and_sabato_2007()
variables_paraschiv_2021          = get_variables_paraschiv_2021()

do_oversampled_data = FALSE

if(do_oversampled_data==FALSE){
  # Load imputed data
  data = read.csv('../data_4_imputed/data_imputed.csv',sep=';')
  
  # Extracting only variables needed, to save memory
  data = data[,get_all_variables()]
  
  results_folder = 'results'
}else{
  results_folder = 'results_sampled_data'
}

variable_sets = c(
  'altman_1968',
  'altman_and_sabato_2007',
  'paraschiv_2021',
  'altman_1968_and_non_financials',
  'altman_and_sabato_2007_and_non_financials',
  'paraschiv_2021_and_non_financials',
  'only_non_financials'
)

bankrupt_variables = c(
  'bankrupt_1',
  'bankrupt_2',
  'bankrupt_3'
)


for (bankrupt_variable in bankrupt_variables){
  start_time_bankrupt_variable = Sys.time()
  
  if(do_oversampled_data){
    data = read.csv(paste("../data_5_sampled/data_sampled_", bankrupt_variable, ".csv", sep = ""),sep=';')
  }
  
  
  first_year = 2018
  last_year = 2019
  
  for (variable_set in variable_sets){
    start_time_variable_set = Sys.time()
    
    
    if(variable_set=='only_non_financials'){
      variables = variables_non_financial
    }else if(variable_set=='altman_1968'){
      variables = variables_altman_1968
    }else if(variable_set=='altman_and_sabato_2007'){
      variables = variables_altman_and_sabato_2007
    }else if(variable_set=='paraschiv_2021'){
      variables = variables_paraschiv_2021
    }else if(variable_set=='altman_1968_and_non_financials'){
      variables = c(variables_non_financial,variables_altman_1968)
    }else if(variable_set=='altman_and_sabato_2007_and_non_financials'){
      variables = c(variables_non_financial,variables_altman_and_sabato_2007)
    }else if(variable_set=='paraschiv_2021_and_non_financials'){
      variables = c(variables_non_financial,variables_paraschiv_2021)
    }else if(variable_set=='all_financial_and_non_financials'){
      variables = unique(c(variables_non_financial,variables_altman_1968,variables_altman_and_sabato_2007,variables_paraschiv_2021))
    }else if(variable_set=='only_financials'){
      variables = unique(c(variables_altman_1968,variables_altman_and_sabato_2007,variables_paraschiv_2021))
    }else{
      print('ERROR defining variable_set')
    }
    
    for (year in first_year:last_year){
      start_time_year = Sys.time()
      
      # Defining training data
      data_train  = as.matrix(data[((data$regnaar<year)&(data$regnaar>=(year-4))),c(variables,bankrupt_variable)])
      X_train = data_train[,variables]
      y_train = data_train[,bankrupt_variable]
      
      # Defining test data
      data_test   = as.matrix(data[(data$regnaar==year),c(variables,bankrupt_variable)])
      X_test  = data_test[,variables]
      y_test  = data_test[,bankrupt_variable]
      
      # Extract last ten characters
      check_if_non_financials = substr(variable_set, nchar(variable_set) - 14 + 1, nchar(variable_set)) 
      
      if((check_if_non_financials=='non_financials')|(variable_set=='only_financials')){
        # Select with LASSO if including non-financial variables
        
        # Tuning lambda value
        set.seed(1)
        cvfit_train = cv.glmnet(X_train, y_train, alpha=1, family = "binomial", nfolds = 10, type.measure = "auc",standardize = TRUE)
        lambda_1se = cvfit_train$lambda.1se
        
        # Training LASSO with tuned lambda value
        glm.model = glmnet(X_train, y_train, alpha=1, family = "binomial", lambda = lambda_1se, standardize = TRUE)
        
        # Extracting selected variables
        variables_selected = as.matrix(coef(glm.model))
        variables_selected = rownames(variables_selected)[variables_selected!=0]
        variables_selected = variables_selected[!variables_selected == '(Intercept)']
  
        ##################################
        ## Make and save LASSO path
        ##################################
        
        # LASSO over a grid
        grid = lambda_1se*10^seq(6,0, length = 200)
        lasso_path = glmnet(X_train[,variables_selected], y_train, alpha = 1, lambda = grid, family = 'binomial',standardize = TRUE)
        
        # Extracting values for LASSO path
        beta = data.frame(t(as.matrix(lasso_path$beta)))
        rownames(beta) = as.character(grid)
        
        # Sorting based on what variables became non-zero first
        temp = data.frame(non_zeros=colSums(beta==0),variable=colnames(beta))
        temp = arrange(temp, non_zeros)
        temp = temp$variable
        beta = beta[,temp]
        
        # Remove rows with all zero coefficients except the first one
        temp = sum(rowSums(beta==0)==ncol(beta))
        beta = beta[temp:nrow(beta),]
        
        # Converting data frame for plotting with ggplot
        beta = data.frame(x = rownames(beta),beta)
        beta = melt(beta, id.vars = "x")
        beta$x = as.numeric(beta$x)
        
        # Renaming variables to correct variable names
        temp = rename_variables_to_correct_text(as.character(beta$variable))
        beta$variable = factor(temp,levels=unique(temp))
    
        # Making plot
        text_size_plot = 18
        if (variable_set=='paraschiv_2021_and_non_financials'){
          text_size_legend = 14
        } else {
          text_size_legend = text_size_plot
        }
        plot_LASSO_path = ggplot(beta, aes(x=x, y=value, color=variable)) +
          geom_line(size=1) + 
          xlab(expression(lambda)) + 
          ylab('Standardized coefficients') +
          scale_x_reverse() + # Reverse x axis
          theme(legend.title = element_blank()) + # Blank legend title
          # theme(legend.position = c(0.26, 0.20)) +
          theme(legend.position = 'bottom') +
          guides(color=guide_legend(nrow=ceiling(length(unique(beta$variable))/2), byrow=FALSE))+
          theme(text=element_text(size=text_size_plot), #change font size of all text
                axis.text=element_text(size=text_size_plot, family="Times New Roman"), #change font size of axis text
                axis.title=element_text(size=text_size_plot, family="Times New Roman"), #change font size of axis titles
                plot.title=element_text(size=text_size_plot), #change font size of plot title
                # legend.title=element_text(size=text_size_plot), #change font size of legend title
                legend.text=element_text(size=text_size_legend, family="Times New Roman")) #change font size of legend text
        
        # Save plot to file
        filename = paste(results_folder,'/plots_LASSO_path/',variable_set,'_',as.character(year),'_',bankrupt_variable,'.png',sep='')
        ggsave(filename, width = 25, height = 16, units = "cm")
        
    
      }else{
        # If using only financial variables, use all
        variables_selected = variables
      }
      
      # Making model formula
      formula_string = paste(variables_selected, sep=" ", " + ", collapse = " ")
      formula_string = paste(bankrupt_variable,"~", formula_string)
      formula_string = substr(formula_string, 1, nchar(formula_string)-2) # Remove last +
      
      # Logistic regression
      lrmod = glm(formula_string, family=binomial(link = "logit"),data = data.frame(data_train))
      
      # Getting coefficients
      coefs = data.frame('variable'=rownames(data.frame(coef(lrmod))),temp=coef(lrmod))
      coefs = rbind(coefs[-1, ], coefs[1, ]) # Moving intercept last
      
      # Getting z-scores
      zscores = summary(lrmod)$coefficients[,3]
      zscores = data.frame('variable'=rownames(data.frame(zscores)),temp=zscores)
      zscores = rbind(zscores[-1, ], zscores[1, ]) # Moving intercept last
      
      # Formatting numbers and putting z-scores in parentheses
      for(i in coefs$variable){
        char1 = as.character(round(as.numeric(coefs[coefs$variable==i,'temp']),digits = 2))
        char2 = as.character(round(as.numeric(zscores[zscores$variable==i,'temp']),digits = 2))
        
        # Adding zero decimals
        if (grepl('.',char1, fixed=T)==FALSE){
          char1 = paste(char1,'.0',sep='')
        }
        
        iii = 0
        while ((str_count(strsplit(char1,'.',fixed=T)[[1]][2])<2)&(iii<10)){
          char1 = paste(char1,0,sep='')
          iii=iii+1
        }
        
        # Adding zero decimals
        if (grepl('.',char2, fixed=T)==FALSE){
          char2 = paste(char2,'.0',sep='')
        }
        
        iii = 0
        while ((str_count(strsplit(char2,'.',fixed=T)[[1]][2])<2)&(iii<10)){
          char2 = paste(char2,0,sep='')
          iii=iii+1
        }
        
        coefs[coefs$variable==i,'temp'] = paste(char1,'(',char2,')',sep='')
      }
      
      # Renaming to current testing year
      colnames(coefs)[colnames(coefs) == 'temp'] = as.character(year)
      
      # Adding coefficients to data.frame for results
      if ((year==first_year)&(variable_set==variable_sets[[1]])){
        coef_df = coefs
      }else{
        coef_df = merge(coef_df, coefs, by='variable', all=TRUE, sort=FALSE)
      }

      
      # McFaddens R squared
      metrics_names = 'McFaddens_R'
      metrics = as.numeric(pscl::pR2(lrmod)["McFadden"])
      
      k   = length(coef(lrmod))
      
      # In-sample
      probs     = predict(lrmod, data.frame(X_train),type = 'response')
      pre_name  = "In-sample"
      temp          = evaluate_model_metrics(metrics, metrics_names, probs, k, pre_name, y_train)
      metrics_names = temp$metrics_names
      metrics       = temp$metrics
      
      # Out-of-sample
      probs     = predict(lrmod, data.frame(X_test),type = 'response')
      pre_name  = "Out-of-sample"
      temp          = evaluate_model_metrics(metrics, metrics_names, probs, k, pre_name, y_test)
      metrics_names = temp$metrics_names
      metrics       = temp$metrics
      
      # Formatting numbers
      metrics_year = data.frame('variable'=metrics_names,temp=metrics)

      colnames(metrics_year)[colnames(metrics_year) == 'temp'] = as.character(year)
      
      # Adding coefficients to data.frame for results
      if ((year==first_year)&(variable_set==variable_sets[[1]])){
        metrics_df = metrics_year
      }else{
        metrics_df = merge(metrics_df, metrics_year, by='variable', all=TRUE, sort=FALSE)
      }
      
      print(paste(bankrupt_variable,' - ',variable_set,' - ',year,': ',round(difftime(Sys.time(), start_time_year, units = "mins"),2),' minutes',sep=''))
    }
    print(paste(bankrupt_variable,' - ',variable_set,': ',round(difftime(Sys.time(), start_time_bankrupt_variable, units = "mins"),2),' minutes',sep=''))
  }
  ##################################
  ## Merging to one table
  ##################################
  
  #set specific column as row names
  rownames(coef_df) <- coef_df$variable
  rownames(metrics_df) <- metrics_df$variable
  
  #remove original column from data frame
  coef_df$variable <- NULL
  metrics_df$variable <- NULL
  
  # Sorting list of coefficients
  temp_rows_before = length(rownames(coef_df))
  list_for_sorting = c(get_all_variables(),'(Intercept)')
  list_for_sorting = list_for_sorting[list_for_sorting %in%  rownames(coef_df)]
  coef_df = coef_df[list_for_sorting,]
  if(temp_rows_before != length(rownames(coef_df))){
    print(paste('ERROR when sorting list of coefficients. Difference:',temp_rows_before-length(rownames(coef_df))))
  }
  
  # Merging coefficients and metric values
  results_df = rbind(coef_df,metrics_df)
  
  # Rename to no brackets
  row.names(results_df)[row.names(results_df) == '(Intercept)'] = 'Intercept'
  
  # Renaming variables to correct variable names
  row.names(results_df) = rename_variables_to_correct_text(row.names(results_df))
  
  # Save to Excel
  dir.create(results_folder)
  filename = paste(paste(results_folder,'\\',bankrupt_variable,sep=''),'.xlsx',sep='')
  write.xlsx(results_df, filename,showNA = FALSE)
  
  print(paste(bankrupt_variable,': ',round(difftime(Sys.time(), start_time_variable_set, units = "mins"),2),' minutes',sep=''))
}

