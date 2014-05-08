# Run Random Forest Model -----------------------------------------------------------------------
source('D:/WK/R&D/Random_Forest/rScript/Random_Forest.R')

if(!exists("rf1")){
  rf1 <- rf_model1(data = data, train_size = train_size,
                   var_remove = var_remove, var_clus = var_clus, var_selected = var_selected, 
                   rf_ntree = rf_ntree, rf_mtry = rf_mtry)
}

# Graphical Parameters ---------------------------------------------------------------------------
# Hardcoded Graphical Parameters
# Temporary only
clrs <- c("#E69F00", "#56B4E9", "#8B4513", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#551A8B", "#999999")
fs <- 20
fs.sub <- 4
cs <- 6
cs.sub <- 3
ts <- 30
ts.pdf <- 20

# Data preparation for Graphics(ready to use for graphics)----------------------------------------
# Converts confusion into Accuracy
rf_accuracy <- function(confusion){
  class_accuracy <- data.frame(1- confusion[,'class.error']) # Converting Class Error --> Accuracy
  all_accuracy <- sum(diag(confusion))/sum(confusion)        # All level Accuracy
  accuracy <- rbind(all_accuracy, class_accuracy)            # Combining All & Class Accuracy
  rownames(accuracy)[1] <- "All"
  accuracy <- round(accuracy*100, 1)                         # Rounding Accuracy
  accuracy <- data.frame(rownames(accuracy), accuracy)
  colnames(accuracy) <- c("Segment", "Accuracy")
  return(accuracy)
}

# Converts confusion into confusion table
rf_confusion_table <- function(confusion){
  confusion <- rf1$confusion
  confusion[,'class.error'] <- round((1 - confusion[,'class.error'])*100, 1) # Converting Error --> Accuracy
  confusion <- data.frame(rownames(confusion), confusion)             # Adding 'Class' names column
  colnames(confusion)[] <- c("Segment", rownames(confusion), "Accuracy")     # Renaming 'Columns'
  
  confusion <- melt(confusion, id=1)
  names(confusion) <- c("Y","X","Count")
  return(confusion)
}

# Importance
rf_Importance <- function(importance){
  importance <- data.frame(rownames(importance), round(importance, 2))
  names(importance)[c(1, ncol(importance)-1, ncol(importance))] <- c("Predictor", "mda", "mdg")
  rownames(importance) <- NULL
  return(importance)
}

# Proximity Data
rf_prox_data <- function(rf_model){
  mds <- cmdscale(1 - rf_model$proximity, eig=TRUE) ## Do MDS on 1 - proximity 
  d <- data.frame(mds$points, # Dim_1, Dim_2
                  rf_model$y, # Classes
                  mds$eig, # Eigen Values
                  as.numeric(margin(rf_model)), # Margin
                  outlier(rf_model)) # Outliers
  names(d) <- c("Dim.1","Dim.2", "Class", "Eigen values", "Margin", "Outliers")
  return(d)
}

# Data Prepare for Graphics ----------------------------------------------------------------------
accuracy_train <- rf_accuracy(rf1$confusion)
accuracy_test <- rf_accuracy(rf1$test$confusion)
confusion_table_train <- rf_confusion_table(rf1$confusion)
confusion_table_test <- rf_confusion_table(rf1$test$confusion)
importance_data <- rf_Importance(importance(rf1))
prox_data <- rf_prox_data(rf1)


# Graphical Plots for Random Forest Analysis------------------------------------------------------

# Classification Accuracy for each Class/Segment. All level accuracy is provided
# Used for both Test & Train Classification Accuracy
classAccuracyPlot <- function(accuracy, confusion_table, celltextsize, fontsize, 
                              titlesize, my_title){
  
  fontsize <- as.numeric(fontsize)
  titlesize <- as.numeric(titlesize)
  g1 <- ggplot(data=accuracy,aes(x=Segment, y=Accuracy, group=Segment, colour=Segment, fill=Segment)) +
    theme_grey(base_size=fontsize) + 
    theme(legend.position="top", legend.key.width=unit(0.1/nlevels(accuracy$Segment),"npc")) +
    scale_fill_manual(values=c(clrs[1:(nlevels(accuracy$Segment))])) + 
    scale_colour_manual(values=c(clrs[1:(nlevels(accuracy$Segment))])) +
    geom_bar(stat="identity") +
    geom_text(aes(label=paste0(Accuracy, "%")), colour="white", vjust=2, size=fontsize/3) +
    scale_y_continuous() + labs(title = my_title)
  g1 <- g1 + theme(plot.title=element_text(size=titlesize, colour = clrs[6]))
  
  g2 <- ggplot(data=confusion_table, aes(x = X, y = Y, fill = Count, label=Count)) + 
    theme_grey(base_size=fontsize) + theme(legend.position="top", legend.key.width=unit(0.1,"npc")) +
    labs(x = "Predicted Segments and Segment Accuracy", y = "Original Segment") +
    geom_raster() +
    scale_fill_gradient( low = "white", high = "dodgerblue", na.value="black", name = "Correct Classification(Count)" ) +
    geom_text(size=celltextsize) +
    geom_rect(size=0.5, fill=NA, colour="black",
              aes(xmin=length(levels(X))-0.5, xmax=length(levels(X))+0.5, ymin=1-0.5, ymax=length(levels(Y))+0.5)) +
    geom_rect(size=0.5, fill=NA, colour="black",
              aes(xmin=1-0.5, xmax=length(levels(X))+0.5, ymin=1-0.5, ymax=length(levels(Y))+0.5)) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0))
  g2
  
  gA <- ggplot_gtable(ggplot_build(g1))
  gB <- ggplot_gtable(ggplot_build(g2))
  gA$widths <- gB$widths
  grid.arrange(gA,gB,ncol=1)
}

# Variable Importance for overall Classification
# Default Mean Decrease in Gini ('mdg') is used
importancePlot <- function(d, ylb, fontsize, titlesize, my_title){
  fontsize <- as.numeric(fontsize)
  titlesize <- as.numeric(titlesize)
  d <- d[order(d[,ylb]),]
  d$Predictor <- factor(as.character(d$Predictor),levels=rev(as.character(d$Predictor)))
  rownames(d) <- NULL
  abs.min <- abs(min(d[,2]))
  g1 <- ggplot(data=d,aes_string(x="Predictor",y=ylb,group="Predictor",colour="Predictor",fill="Predictor")) + geom_bar(stat="identity") + theme_grey(base_size=fontsize)
  if(ylb=="mda") g1 <- g1 + labs(y="Mean decrease in accuracy") else if(ylb=="mdg") g1 <- g1 + labs(y="Mean decrease in Gini")
  g1 <- g1 + theme(axis.text.x = element_text(angle=90,hjust=1,vjust=0.4)) + geom_hline(yintercept=abs.min,linetype="dashed",colour="black")
  g1 <- g1 + labs(title = my_title)
  g1 <- g1 + theme(plot.title=element_text(size=titlesize, colour = clrs[6]))
  print(g1)
}

# Variable Importance by Class/Segment along with variable importance at overall level i.e.,
# Mean Decrease in Gini ('mdg') & Mean Decrease in Accuracy ('mda')
importanceTable <- function(d,lab, celltextsize, fontsize, titlesize, my_title){
  d <- melt(d)
  names(d) <- c("Y","X","Importance")
  
  fontsize <- as.numeric(fontsize)
  titlesize <- as.numeric(titlesize)
  g1 <- ggplot(data=d, aes(x = X, y = Y, fill = Importance, label=Importance)) + 
    theme_grey(base_size=fontsize) + 
    theme(legend.position="top", legend.key.width=unit(0.1,"npc")) +
    labs(x = "Response class and mean performance measures", y = "Predictor") +
    geom_raster() +
    geom_text(size=celltextsize) +
    scale_fill_gradient( low = "white", high = "dodgerblue", na.value="black", name = "Importance" ) +
    geom_rect(size=1, fill=NA, colour="black",
              aes(xmin=length(levels(X))-1-0.5, xmax=length(levels(X))-1+0.5, ymin=1-0.5, ymax=length(levels(Y))+0.5)) +
    geom_rect(size=2, fill=NA, colour="black",
              aes(xmin=1-0.5, xmax=length(levels(X))+0.5, ymin=1-0.5, ymax=length(levels(Y))+0.5)) +
    scale_x_discrete(expand = c(0, 0),labels=lab) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(axis.text.x = element_text(hjust=1,vjust=0.4)) + labs(title = my_title)
  g1 <- g1 + theme(plot.title=element_text(size=titlesize, colour = clrs[6]))
  print(g1)
}

# Multi Dimensional Scaling 
mdsPlot <- function(d, fontsize, titlesize, my_title){
  fontsize <- as.numeric(fontsize)
  titlesize <- as.numeric(titlesize)
  g1 <- ggplot(data=d, aes_string(x=names(d)[1], y=names(d)[2], group=names(d)[3], colour=names(d)[3])) + 
    theme_grey(base_size=fontsize) + theme(legend.position="top") +
    scale_fill_manual(values=c(clrs[1:(nlevels(d[,3]))])) + 
    scale_colour_manual(values=c(clrs[1:(nlevels(d[,3]))])) +
    geom_point(size=3) + labs(title = my_title)
  g1 <- g1 + theme(plot.title=element_text(size=titlesize, colour = clrs[6]))
  print(g1)
}

# Plots ---------------
classAccuracyPlot(accuracy = accuracy_train, confusion_table = confusion_table_train, 
                  celltextsize = cs, fontsize = fs, titlesize = ts,
                  my_title = "Classification Accuracy (Training Data)")
importancePlot(d = importance_data, ylb = "mdg", fontsize = fs, titlesize = ts,
               my_title = "Variable Importance")
importanceTable(d = importance_data, lab = colnames(importance_data)[-1], 
                celltextsize =cs, fontsize = fs, titlesize = ts,
                my_title = "Variable Importance")
mdsPlot(d = prox_data, fontsize = fs, titlesize = ts,
        my_title = "MDS Plot")
