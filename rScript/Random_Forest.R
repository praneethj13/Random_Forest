# Load libraries -------------------------------------------------------------------
library(randomForest) # Non-Parametric Model
library(ggplot2)   # Main Graphics
library(caret)     # Main Analysis
library(grid)      # Support for Graphics
library(gridExtra) # Support for Graphics
library(scales)    # Support for Graphics
library(reshape2)  # Data Transformation
library(plyr)      # Data Transformation

# Libraries for Parallel Computation
library(parallel)
library(doParallel)
library(foreach)

# Data & Model Parameters ----------------------------------------------------------
# Hardcoded Model Parameters
# Temporary only
data <- read.csv("data/Segmentation_Data.csv") # Loading data
train_size <- 0.7 # Size of Training Set
var_remove <- c("ID") # Variables not considered in modeling
var_clus <- "Cluster" # Class or Cluster Variable (alias predictor)
var_selected <- NULL # Variables to be considered in model
rf_ntree <- 400 # no. of trees to be used in the model
rf_mtry  <- 5 # no. of variables used in each tree

# Random Forest Models --------------------------------------------------------------
# 
# Model 1 : Simple Non-Parallel Model -------------------------------------
# Calculating Training Rows 
train_rows <- function(data, train_size){
  x <- createDataPartition(data, p = train_size, list = FALSE)
  return(x)
}


rf_model1 <- function(data = data, train_size = train_size,
                      var_remove = var_remove, var_clus = var_clus, var_selected = var_selected, 
                      rf_ntree = rf_ntree, rf_mtry = rf_mtry){
  
  # Removing Variables (Will be selected by User)
  # ? This can be moved out of rf_model1 function
  data <- data[, -which(names(data) %in% var_remove)] 
  
  # Selected Variables (Will be selected by User)
  # ? This can be moved out of rf_model1 function
  if(!is.null(var_selected)) {
    var_selected1 <- c(var_selected, var_clus)
    data <- data[, which(names(data) %in% var_selected)] 
  }
  
  # Training & Test Set Data 
  train_rows <- train_rows(data = data[, var_clus], train_size = train_size)
  y_train <- data[train_rows, var_clus]
  x_train <- data[train_rows, -which(names(data) %in% var_clus)] 
  y_test <- data[-train_rows, var_clus]
  x_test <- data[-train_rows, -which(names(data) %in% var_clus)] 
  
  # Notes on Random Forest Speed
  # 30% More Time for Importance Calculation
  # 50% More Time for Proximity Calculation
  # Adding Test Set data, doesn't effect the speed of Random Forest much
  # 3x speed achieved with random forest in parallel execution (4 core computer) 
  # But including Importance, Proximity & OOB.proximity normal is 1.3x faster than parallel version
  
  # n_cores <- detectCores() # Detects cores
  n_cores <- 1
  # If more than 1 (usually 4 in latest PC's) runs random forest in parallel mode or else in normal mode
  if(n_cores > 1){
    # Created clusters for parallel execution
    cl <- makePSOCKcluster(n_cores)
    registerDoParallel(cl)
    
    # Random forest in parallel mode
    rf_ntree <- ceiling(rf_ntree/n_cores)
    rf_model_parallel <- foreach(ntree = rep(rf_ntree, n_cores), .combine = combine, 
                                 .multicombine = TRUE, .packages = 'randomForest') %dopar% {
                                   randomForest(x = x_train, y = y_train, xtest = x_test, ytest = y_test,
                                                ntree = rf_ntree, mtry = rf_mtry, importance = TRUE)
                                 }
    stopCluster(cl) 
    return(rf_model_parallel) # Return random forest object
    
    } else if(n_cores==1){
    # Random forest in normal mode
    rf_model_normal <- randomForest(x = x_train, y = y_train, xtest = x_test, ytest = y_test,
                                    ntree = rf_ntree, mtry = rf_mtry, importance = TRUE, proximity = TRUE,
                                    keep.forest = TRUE)
    # Adding data to the model
    ln <- length(rf_model_normal)
    rf_model_normal[[ln+1]] <- data
    rf_model_normal[[ln+2]] <- data[, -which(names(data) %in% var_clus)]
    names(rf_model_normal)[(ln+1):(ln+2)] <- c("org_data", "org_data_wo_class")
    
    return(rf_model_normal) # Return random forest object
  }
    
}

