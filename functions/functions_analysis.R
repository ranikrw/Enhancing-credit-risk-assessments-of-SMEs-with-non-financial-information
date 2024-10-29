
rename_variables <- function(variables) {
  for(i in 1:length(variables)) {
    variables[i] = gsub(" ", ".", variables[i])
    variables[i] = gsub("/", ".", variables[i])
    variables[i] = gsub("-", ".", variables[i])
    variables[i] = gsub("[(]", ".", variables[i])
    variables[i] = gsub("[)]", ".", variables[i])
    variables[i] = gsub(";", ".", variables[i])
    if (substr(variables[i], 1, 1)=="."){
      variables[i] = paste('X',variables[i],sep="")
    }
  }
  return(variables)
}

get_variables_non_financial <- function() {
  variables_non_financial = c(
    'CEO age',
    'CEO woman',
    'CEO duality',
    'CEO on board',
    'CEO county',
    'two CEOs',
    'chairperson age',
    'chairperson woman',
    'chairperson county',
    'board size',
    'board age avg',
    'board age std',
    'board women',
    'board county',
    'board non-owners',
    'ownership concentration 1',
    'ownership concentration 2',
    'ownership CEO',
    'ownership chairperson',
    'ownership board'
  )
  variables_non_financial = rename_variables(variables_non_financial)
  return(variables_non_financial)
}


get_variables_altman_1968 <- function() {
  variables_altman_1968 = c(
    'EBIT / total assets',
    'retained earnings / total assets',
    'sales / total assets',
    'total equity / total liabilities',
    'working capital / total assets'
  )
  variables_altman_1968 = rename_variables(variables_altman_1968)
  return(variables_altman_1968)
}

get_variables_altman_and_sabato_2007 <- function() {
  variables_altman_and_sabato_2007 = c(
    'current liabilities / total equity',
    'EBITDA / interest expense',
    'EBITDA / total assets',
    'retained earnings / total assets',
    'short-term liquidity / total assets'
  )
  variables_altman_and_sabato_2007 = rename_variables(variables_altman_and_sabato_2007)
  return(variables_altman_and_sabato_2007)
}


get_variables_paraschiv_2021 <- function() {
  variables_paraschiv_2021 = c(
    '(current liabilities - short-term liquidity) / total assets',
    'accounts payable / total assets',
    'dummy; one if paid-in equity is less than total equity',
    'dummy; one if total liability exceeds total assets',
    'interest expenses / total assets',
    'inventory / current assets',
    'log(age in years)',
    'net income / total assets',
    'public taxes payable / total assets',
    'short-term liquidity / current assets'
  )
  variables_paraschiv_2021 = rename_variables(variables_paraschiv_2021)
  return(variables_paraschiv_2021)
}

get_all_variables <- function() {
  all_variables = unique(c(
    get_variables_non_financial(),
    get_variables_altman_1968(),
    get_variables_altman_and_sabato_2007(),
    get_variables_paraschiv_2021(),
    'bankrupt_fs',
    'bankrupt_1',
    'bankrupt_2',
    'bankrupt_3',
    'bankrupt_4',
    'bankrupt_5',
    'regnaar'
  ))
  return(all_variables)
}


rename_variables_to_correct_text <- function(vec) {
  for(i in 1:length(vec)){
    if(vec[i]=='CEO.age'){
      vec[i]='CEO age'
    }else if(vec[i]=='CEO.woman'){
      vec[i]='CEO woman'
    }else if(vec[i]=='CEO.duality'){
      vec[i]='CEO duality'
    }else if(vec[i]=='CEO.on.board'){
      vec[i]='CEO on board'
    }else if(vec[i]=='CEO.county'){
      vec[i]='CEO county'
    }else if(vec[i]=='two.CEOs'){
      vec[i]='two CEOs'
    }else if(vec[i]=='chairperson.age'){
      vec[i]='chairperson age'
    }else if(vec[i]=='chairperson.woman'){
      vec[i]='chairperson woman'
    }else if(vec[i]=='chairperson.county'){
      vec[i]='chairperson county'
    }else if(vec[i]=='board.size'){
      vec[i]='board size'
    }else if(vec[i]=='board.age.avg'){
      vec[i]='board age avg'
    }else if(vec[i]=='board.age.std'){
      vec[i]='board age std'
    }else if(vec[i]=='board.women'){
      vec[i]='board women'
    }else if(vec[i]=='board.county'){
      vec[i]='board county'
    }else if(vec[i]=='board.non.owners'){
      vec[i]='board non-owners'
    }else if(vec[i]=='ownership.concentration.1'){
      vec[i]='ownership concentration 1'
    }else if(vec[i]=='ownership.concentration.2'){
      vec[i]='ownership concentration 2'
    }else if(vec[i]=='ownership.CEO'){
      vec[i]='ownership CEO'
    }else if(vec[i]=='ownership.chairperson'){
      vec[i]='ownership chairperson'
    }else if(vec[i]=='ownership.board'){
      vec[i]='ownership board'
    }else if(vec[i]=='EBIT...total.assets'){
      vec[i]='EBIT / total assets'
    }else if(vec[i]=='retained.earnings...total.assets'){
      vec[i]='retained earnings / total assets'
    }else if(vec[i]=='sales...total.assets'){
      vec[i]='sales / total assets'
    }else if(vec[i]=='total.equity...total.liabilities'){
      vec[i]='total equity / total liabilities'
    }else if(vec[i]=='working.capital...total.assets'){
      vec[i]='working capital / total assets'
    }else if(vec[i]=='current.liabilities...total.equity'){
      vec[i]='current liabilities / total equity'
    }else if(vec[i]=='EBITDA...interest.expense'){
      vec[i]='EBITDA / interest expense'
    }else if(vec[i]=='EBITDA...total.assets'){
      vec[i]='EBITDA / total assets'
    }else if(vec[i]=='retained.earnings...total.assets'){
      vec[i]='retained earnings / total assets'
    }else if(vec[i]=='short.term.liquidity...total.assets'){
      vec[i]='short-term liquidity / total assets'
    }else if(vec[i]=='X.current.liabilities...short.term.liquidity....total.assets'){
      vec[i]='(current liabilities - short-term liquidity) / total assets'
    }else if(vec[i]=='accounts.payable...total.assets'){
      vec[i]='accounts payable / total assets'
    }else if(vec[i]=='dummy..one.if.paid.in.equity.is.less.than.total.equity'){
      vec[i]='dummy; one if paid-in equity is less than total equity'
    }else if(vec[i]=='dummy..one.if.total.liability.exceeds.total.assets'){
      vec[i]='dummy; one if total liability exceeds total assets'
    }else if(vec[i]=='interest.expenses...total.assets'){
      vec[i]='interest expenses / total assets'
    }else if(vec[i]=='inventory...current.assets'){
      vec[i]='inventory / current assets'
    }else if(vec[i]=='log.age.in.years.'){
      vec[i]='log(age in years)'
    }else if(vec[i]=='net.income...total.assets'){
      vec[i]='net income / total assets'
    }else if(vec[i]=='public.taxes.payable...total.assets'){
      vec[i]='public taxes payable / total assets'
    }else if(vec[i]=='short.term.liquidity...current.assets'){
      vec[i]='short-term liquidity / current assets'
    }else if(vec[i]=='McFaddens_R'){
      vec[i]='R2'
    }else if(vec[i]=='AUC_in_sample'){
      vec[i]='In-sample AUC'
    }else if(vec[i]=='Brierscore_in_sample'){
      vec[i]='In-sample Brier score'
    }else if(vec[i]=='AUC_out_of_sample'){
      vec[i]='Out-of-sample AUC'
    }else if(vec[i]=='Brierscore_out_of_sample'){
      vec[i]='Out-of-sample Brier score'
    }
  }
  return(vec)
}


calculate_ar <- function(y_true, y_pred) {
  # Create a data frame and sort by predicted probabilities
  data <- data.frame(y_true = y_true, y_pred = y_pred)
  data <- data[order(-data$y_pred), ]
  
  # Calculate cumulative true positives and total positives
  data$cum_true_positives <- cumsum(data$y_true)
  total_positives <- sum(data$y_true)
  
  # Calculate cumulative percentage of positives and population
  data$cum_percentage_positives <- data$cum_true_positives / total_positives
  data$cum_percentage_population <- seq_along(data$y_true) / length(data$y_true)
  
  # Calculate areas
  area_model <- DescTools::AUC(data$cum_percentage_population, data$cum_percentage_positives)
  area_random <- 0.5
  area_perfect <- 1.0
  
  # Calculate Accuracy Ratio (AR)
  ar <- (area_model - area_random) / (area_perfect - area_random)
  return(ar)
}

HingeLoss <- function(y_true, y_pred) {
  return(mean(pmax(0, 1 - y_true * y_pred)))
}

make_decile_rankings <- function(pred, y_test) {
  num_deciles <- 10
  num_last_deciles_to_group <- 5
  
  temp <- data.frame(targets = y_test, predicted = pred)
  temp <- temp[order(-temp$predicted), ]
  temp$decile <- cut(temp$predicted, breaks = quantile(temp$predicted, probs = seq(0, 1, length.out = num_deciles + 1)), 
                     labels = rev(1:num_deciles), include.lowest = TRUE)
  
  decile_rankings <- c()
  for (i in 1:(num_deciles - num_last_deciles_to_group)) {
    decile_rankings[paste("Decile", i)] <- sum(temp[temp$decile == i, "targets"]) / sum(temp$targets)
  }
  
  deciles_last <- (num_deciles - num_last_deciles_to_group + 1):num_deciles
  decile_rankings[paste("Decile ", min(deciles_last), "-", max(deciles_last), sep = "")] <- 
    sum(temp[temp$decile %in% deciles_last, "targets"]) / sum(temp$targets)
  
  return(decile_rankings)
}

evaluate_model_metrics <- function(metrics, metrics_names, probs, k, pre_name, y_train) {
  # Append new metric names
  new_metrics_names <- c('AUC', 'AP', 'AR', 'KS statistic', 'Hinge Loss', 'Log loss', 'AIC', 'BIC', 'Brier score', 'BSS')
  metrics_names <- c(metrics_names, paste(pre_name, new_metrics_names))
  
  # Calculate metrics
  new_metrics <- c(
    pROC::auc(pROC::roc(y_train, probs)),  # AUC
    PRAUC(probs, y_train),  # AP
    calculate_ar(y_train, probs),  # AR
    ks.test(probs[y_train == 1], probs[y_train == 0])$statistic,  # KS statistic
    HingeLoss(ifelse(y_train == 1, 1, -1), probs),  # Hinge Loss
    logLoss(y_train, probs),  # Log loss
    2 * k + 2 * logLoss(y_train, probs)*length(y_train),  # AIC
    k * log(length(probs)) + 2 * logLoss(y_train, probs)*length(y_train),  # BIC
    BrierScore(y_train, probs),  # Brier score
    1 - (BrierScore(y_train, probs) / (mean(y_train) * (1 - mean(y_train))))  # BSS
  )
  metrics <- c(metrics, new_metrics)
  
  # Add decile rankings to metrics
  decile_rankings   <- make_decile_rankings(probs, y_train)
  metrics_names     <- c(metrics_names, paste(pre_name, names(decile_rankings)))
  metrics           <- c(metrics, decile_rankings)
  
  return(list(metrics = metrics, metrics_names = metrics_names))
}
