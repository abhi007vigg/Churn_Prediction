#Loading the libraries to use in the project
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(readr)
library(gridExtra)
library(corrplot)
library(corrgram)
library(ROCR)
library(data.table)
library(plyr)
library(readr)
library(ranger)
library(pROC)
library(gbm)
#Setting the working directory
setwd("/Users/abhishekvigg/Desktop/Churn")
#Importing data
getwd()
churn <- data.frame(read_csv("churn.csv"))
glimpse(churn)
#Checking for missing values
#Replacing ? with NA in all the columns
col_chars = sapply(churn,is.character)
col_chars = colnames(churn[col_chars])
for (i in col_chars)
{
index = which(churn[,i] == "?")
churn[,i] = replace(churn[,i],index,NA)
}
churn$Eve.Calls = as.numeric(churn$Eve.Calls)
churn$Eve.Mins = as.numeric(churn$Eve.Mins)
churn$Day.Charge = as.numeric(churn$Day.Charge)
churn$Intl.Calls = as.numeric(churn$Intl.Calls)
churn$Intl.Charge = as.numeric(churn$Intl.Charge)
churn$Night.Charge = as.numeric(churn$Night.Charge)
churn$Area.Code = as.numeric(churn$Area.Code)
number_cols = sapply(churn,is.numeric)
number_cols = colnames(churn[number_cols])
for (i in number_cols)
{
  index = which(churn[,i] == '?')
  churn[,i] = replace(churn[,i],index,NA)
}
sum(is.na(churn))
sapply(churn,function(x){sum(is.na(x))})
#Now treating the missing values
#Using median imputation for numeric columns
for (i in number_cols)
{
  index = which(is.na(churn[,i]))
  churn[index,i] = median(churn[,i],na.rm = TRUE)
}
#churn$Int.l.Plan = NULL
#churn$VMail.Plan = NULL
churn = churn[complete.cases(churn),]
#churn_clean = data.frame(churn)
#Replacing yes with 1 and no with 0
#Since a huge number of values are missing in int.l.plan and vmail.plan we will have to delete these variables
#churn = churn %>% mutate(Churn. = ifelse(Churn. == "False.",0,1))
#churn = churn %>% mutate(VMail.Plan = ifelse(VMail.Plan == "no",0,1))
#churn = churn %>% mutate(Int.l.Plan = ifelse(Int.l.Plan == "no",0,1))
churn$Churn. = as.factor(churn$Churn.)
churn$State = as.factor(churn$State)
glimpse(churn)
#State variable has too many classes. hence it is omitted
churn$State = NULL
#churn$State = as.integer(churn$State)
churn$Phone = as.factor(churn$Phone)
churn$Int.l.Plan = as.factor(churn$Int.l.Plan)
churn$VMail.Plan = as.factor(churn$VMail.Plan)
#Running Chi Square Test To Find the significance of the relationship of categorical variables on Churn
col_chars = sapply(churn,is.factor)
col_chars = colnames(churn[col_chars])
categorical_data = churn[,col_chars]
for (i in 1:4)
{
  print(names(categorical_data)[i])
  print(chisq.test(categorical_data$Churn.,categorical_data[,i]))
}
#Phone is a statistically Insignificant Variable in the determination of Churn as its pa value is greater than 0.05
churn$Phone = NULL
churn$X1 = NULL
#Checking for multicollinearity in the numeric variables
numeric_cols = sapply(churn,is.numeric)
numeric_data = churn[,numeric_cols]
#Drawing A Correlation Plot
corr_table = cor(numeric_data)
corrplot(corr_table, method="circle")
#We have multi collinerarity between four pairs of variables. 
correlationMatrix <- cor(numeric_data[,1:16])
#Detecting Variable that are causing high collinearity
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.80)
#Finding Variable names that are to be removed
list_removed = colnames(numeric_data[,highlyCorrelated])
churn[,list_removed] = NULL
#Multicollinearity Is Removed
#Finding variable Importance using gini index from random forest algorithm
set.seed(123)
table(churn$Churn.)
barplot(table(churn$Churn.))
#There is a class imbalance problem with this model
rf_model = randomForest(Churn. ~.,data = churn)
#Variable Importance Plot Continued From Feature Importance
varImpPlot(rf_model,type=2)
#Trying to find the point where tree splits don't decrease error
plot(rf_model)
#Balancing the classes before doing predictive modelling
#over = ovun.sample(Churn.~.,data = churn,method = "over",N = 5700)$data
#Building a Logistic Regression model
#Doing a train test split
sample <- sample.int(n = nrow(churn), size = floor(.75*nrow(churn)), replace = F)
train <- churn[sample, ]
test  <- churn[-sample, ]
#Creating a train control for cross validation
train_control <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)
#train_control <- trainControl(method = "cv", number = 10,classProbs = TRUE)
#Developing Logistic regression model
install.packages('e1071', dependencies=TRUE)
library(e1071)
model <- train(Churn. ~ .,
               data = train,
               trControl = train_control,
               method = "glm",
               family=binomial())
predictions_logistic = predict(model,newdata = test[-16],type = "prob")
#Since we are predicting the churn,we will choose 1 as the target class
predictions_logistic = predictions_logistic[,2]
#Building a baseline model
p = 0.5
#Predicing Classes on the response probabilities
class_predictions = ifelse(predictions_logistic > p,1,0)
conf_logistic_matrix = table(test$Churn.,class_predictions)
conf_logistic_matrix
#A lot of customers that are not supposed to churn are actually classified as churn
#ROCRpred = prediction(predictions_logistic,test$Churn)
#plot(ROCRpred,colorize = TRUE,print.cutoffs.at=seq(0,1,by=0.1),text.adj=c(-0.2,1.7))
library(pROC)
roc_obj <- roc(test$Churn., predictions_logistic)
auc(roc_obj)
#Implementing Random Forest to find a better AUC
tuneGrid <- data.frame(mtry = c(3,8,10,12,13,14),min.node.size = 1,splitrule = c("gini",'extratrees'))
#Fitting The Model
library(ranger)
model_tuned <- train(Churn. ~.,tuneGrid = tuneGrid,data = train, method = "ranger",trControl = train_control)
plot(model_tuned)
predictions_model_tuned = predict(model_tuned,newdata = test[-15],type = "prob")
conf_logistic_matrix = table(test$Churn.,predictions_model_tuned)
conf_logistic_matrix
#Choosing the true class
predictions_model_tuned = predictions_model_tuned[,2]
roc_obj_rf <- roc(test$Churn., predictions_model_tuned)
auc(roc_obj_rf)
summary(model_tuned)
gbm_fit <- train(Churn. ~ .,
                  data = train,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = train_control)
predictions_gbm = predict(gbm_fit,newdata = test[-15],type = "prob")
predictions_gbm = predictions_gbm[,2]
roc_obj_gbm <- roc(test$Churn., predictions_gbm)
auc(roc_obj_gbm)