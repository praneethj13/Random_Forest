# Predicting Class (or Cluster or Segment) of new data -------------------------
# RF Prediction Function

rf_predict <- function(rf_model, new_data){
  
  # Predict Final class & Probability
  pred_class <- predict(rf_model, newdata = new_data, type = "class")
  pred_prob <- predict(rf_model, newdata = new_data, type = "prob")
  
  # Combine with new_data
  data <- cbind(pred_class, pred_prob, new_data)
  
  return(data)
}

# Test prediction function
pred_data <- rf_predict(rf1, rf1$org_data_wo_class)
