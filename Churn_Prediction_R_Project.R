#Importing Libraries In R
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
library(xgboost)
#Setting Up The Working Directory
setwd("E:/Customer_Churn_Edwisor_Project/Customer_Churn_R")
getwd()
#Importing the Train Set
Train_data <- data.frame(read_csv("E:/Customer_Churn_Edwisor_Project/Train_data.csv"))
#Importing Test Data
Test_data <- data.frame(read_csv("E:/Customer_Churn_Edwisor_Project/Test_data.csv"))
#Checking The Types Of Variables And The Data
glimpse(Train_data)
#Data Pre-Processing
#Missing Value Analysis On Train Data
sapply(Train_data,function(x){sum(is.na(x))})
#Missing Value Analysis On Test_Data
sapply(Test_data,function(x){sum(is.na(x))})
#-----There Are No Missing Values In Both Train And Test Data------#
#Outlier Analysis On Data
#Boxplot of Number Vmail Messages
p1 = Train_data %>% ggplot(aes(x = "",y = number.vmail.messages)) + geom_boxplot() + 
ggtitle("Train Set") + xlab("Number Vmail Messages") + ylab("Values")
p2 = Test_data %>% ggplot(aes(x ="",y = number.vmail.messages)) + geom_boxplot() +
ggtitle("Test Data") + xlab("Number Vmail Messages") + ylab("Values")
grid.arrange(p1, p2,ncol=2)
#Boxplot of Total Day Minutes
p3 = Train_data %>% ggplot(aes(x = "",y = total.day.minutes)) + geom_boxplot() + 
ggtitle("Train Set") + xlab("Total Day Minutes") + ylab("Values")
p4 = Test_data %>% ggplot(aes(x = "",y = total.day.minutes)) + geom_boxplot() +
ggtitle("Test Set") + xlab("Total Day Minutes") + ylab("Values")
grid.arrange(p3, p4,ncol=2)
#boxplot of Total Day Calls
p5 = Test_data %>% ggplot(aes(x = "",y = total.day.calls)) + geom_boxplot() +
ggtitle("Test Set") + xlab("Total Day Calls") + ylab("Values")
p6 = Train_data %>% ggplot(aes(x = "",y = total.day.calls)) + geom_boxplot() + 
ggtitle("Train Set") + xlab("Total Day Calls") + ylab("Values")
grid.arrange(p6, p5,ncol=2)
#Feature Selection On Train Data
sapply(Train_data[,7:20],as.numeric)
sapply(Test_data[,7:20],as.numeric)
#Sub Setting the Numeric Data
numeric_cols = sapply(Train_data,is.numeric)
numeric_data = Train_data[,numeric_cols]
#Drawing A Correlation Plot
corr_table = cor(numeric_data)
corrplot(corr_table, method="circle")
#Remove Correlation in numeric variables only after knowing the variable importance
#Chi Square Tests For bivariate 
cat_index = colnames(Train_data[,!numeric_cols])
categorical_data = Train_data[,cat_index]
#Applying Chi Square Tests
for (i in 1:4)
{
  print(names(categorical_data)[i])
  print(chisq.test(categorical_data$Churn,categorical_data[,i]))
}
#Phone Number Has A p-Value Greater Than 0.05
Train_data$phone.number = NULL
Test_data$phone.number = NULL
#Performing Feature Importance Using R to remove multi collinearity and find variable importance
#Finding Correlation
correlationMatrix <- cor(numeric_data[,1:16])
#Detecting Variable that are causing high collinearity
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.80)
#Finding Variable names that are to be removed
list_removed = colnames(numeric_data[,highlyCorrelated])
#We remove the variables in list_removed after performing the exploratory data analysis to prevent multicollinearity in Models
#Exploratory Data Analysis
e1 = Train_data %>% ggplot(aes(x = Churn)) + geom_histogram(stat = "count",fill = "black") + xlab("Response") + ylab("Count Of Response") + ggtitle("Train Churn")
e2 = Test_data %>% ggplot(aes(x = Churn)) + geom_histogram(stat = "count",fill = "black") + xlab("Response") + ylab("Count Of Response") + ggtitle("Test Churn")
grid.arrange(e1,e2,ncol = 2)
#Visualizing the effect of International Plans on Churn
Train_data %>% ggplot(aes(x = Churn,fill = international.plan)) + geom_bar(position = "dodge") + xlab("Churn") + ylab("Count") + ggtitle("Churn Vs International Plan")
#Visualizing the effect of geographic location on churn
top_10 = Train_data %>% filter(Churn == "True.") %>% group_by(state) %>% summarise(Count_Of_Churned_Customers = n()) %>% arrange(desc(Count_Of_Churned_Customers)) %>% top_n(10)
top_10 %>% ggplot(aes(x = state,y = Count_Of_Churned_Customers)) + geom_bar(stat = "sum",fill = "black") + xlab("State name") +ylab("Number Of Churned Customers")
#Checking the effect of Voice Mail Plans On Churn
Train_data %>% ggplot(aes(x = Churn,fill = voice.mail.plan)) + geom_bar(position = "dodge") + xlab("Churn") + ylab("Count") + ggtitle("Churn Vs Voice Mail Plan")
#Checking The effect of Total Charges on Churn
x1 = Train_data %>% group_by(Churn) %>% summarise(Average_Charges = mean(total.day.charge)) %>% ggplot(aes(x = Churn,y = Average_Charges)) + geom_bar(stat = "sum",fill = "red") + ggtitle("Average Day Charges")
x2 = Train_data %>% group_by(Churn) %>% summarise(Average_Charges = mean(total.eve.charge)) %>% ggplot(aes(x = Churn,y = Average_Charges)) + geom_bar(stat = "sum",fill = "blue") + ggtitle("Average Eve Charges")
x3 = Train_data %>% group_by(Churn) %>% summarise(Average_Charges = mean(total.night.charge)) %>% ggplot(aes(x = Churn,y = Average_Charges)) + geom_bar(stat = "sum",fill = "orange") + ggtitle("Average Night Charges")
grid.arrange(x1,x2,x3,ncol = 3)
#Finding the relationship between number of customer service calls and Churn
Train_data %>% group_by(Churn) %>% summarise(Average_Calls = mean(number.customer.service.calls)) %>% ggplot(aes(x = Churn,y = Average_Calls)) + geom_bar(stat = "sum",fill = "red") + ggtitle("Average Customer Calls")
#Model Development
#Checking For Normality Of The Data
n1 = Train_data %>% ggplot(aes(x = total.intl.minutes)) + geom_histogram(stat = "bin")
n2 = Train_data %>% ggplot(aes(x = total.day.calls)) + geom_histogram(stat = "bin")
n3 = Train_data %>% ggplot(aes(x = total.eve.calls)) + geom_histogram(stat = "bin")
#n4 = Train_data %>% ggplot(aes(x = total.intl.calls)) + geom_histogram(stat = "bin")
n4 = Train_data %>% ggplot(aes(x = total.night.charge)) + geom_histogram(stat = "bin")
grid.arrange(n1,n2,n3,n4,ncol = 2)
#Random Forest Model and Its Tuning
#We do not center and scale data for Generalized Regression Models or Tree based Models
#Removing the four variables in list_removed from both train and test data
Train_data[,list_removed] = NULL
Test_data[,list_removed] = NULL
#Recoding a few variables in Train and Test Data
Train_data$Churn = as.factor(Train_data$Churn)
Train_data$state = as.factor(Train_data$state)
Train_data$international.plan = as.factor(Train_data$international.plan)
Train_data$voice.mail.plan = as.factor(Train_data$voice.mail.plan)
Test_data$Churn = as.factor(Test_data$Churn)
Test_data$state = as.factor(Test_data$state)
Test_data$international.plan = as.factor(Test_data$international.plan)
Test_data$voice.mail.plan = as.factor(Test_data$voice.mail.plan)
#Fitting The Random Forest Model
set.seed(123)
rf_model = randomForest(Churn ~.,data = Train_data)
#Variable Importance Plot Continued From Feature Importance
varImpPlot(rf_model,type=2)
#Trying to find the point where tree splits don't decrease error
plot(rf_model)
predictions_rf_model = predict(rf_model,newdata = Test_data)
conf_rf_model = table(Test_data$Churn,predictions_rf_model)
confusionMatrix(conf_rf_model)
#Hypertuning the Random Forest Model With A Custom tunegrid Manipulating mtry for values ranging from 2 to 16
tuneGrid <- data.frame(mtry = c(3,8,10,12,15,16),min.node.size = 1,splitrule = c("gini",'extratrees'))
#Fitting The Model
model_tuned <- train(Churn ~.,tuneGrid = tuneGrid,data = Train_data, method = "ranger",trControl = trainControl(method = "cv", number = 3, verboseIter = TRUE))
#Plotting the hypertuned model
plot(model_tuned)
#Predicting The Responses On Test Data
predictions_model_tuned = predict(model_tuned,newdata = Test_data)
#Tabulating a summary of confusion matrix
conf_model_tuned = table(Test_data$Churn,predictions_model_tuned)
confusionMatrix(conf_model_tuned)
#Logistic Regression Model
#Fitting The Model To Train Data
logistic_model = glm(Churn ~.,data = Train_data,family = "binomial")
#Predicitng the output using the test data
predictions_logistic = predict(logistic_model,newdata = Test_data,type = "response")
#Getting a feel of the class imbalance in our data
grouping_test_data = Test_data %>% group_by(Churn) %>% summarise(Total = n())
#Building a baseline model
p = 0.5
#Predicing Classes on the response probabilities
class_predictions = ifelse(predictions_logistic > p,1,0)
#Making Confusion Matrix to calculate accuracy and false positive rate
conf_logistic_matrix = table(Test_data$Churn,class_predictions)
conf_logistic_matrix
#Sensitivity = 53.92%
55/(55+47)
#Specificity = 89.20%
1396/(1396+169)
#Accuracy = 86.98%
(55+1395)/1667
#Hypertuning
#Our Objective is to predict the customers that actually Churn. #So sensitivity is more important than specificity
#Replacing "True." with 1 and "False." with 0 for the purpose of fitting ROC curve
Test_data_new <- data.frame(read_csv("E:/Customer_Churn_Edwisor_Project/Test_data.csv"))
Test_data_new$Churn[Test_data_new$Churn == "False."] <- "0"
Test_data_new$Churn[Test_data_new$Churn == "True."] <- "1"
ROCRpred = prediction(predictions_logistic,Test_data_new$Churn)
#Performance Function
ROCRperf = performance(ROCRpred, "tpr", "fpr")
#Plotting the sensitivity vs specificity ROC curve
plot(ROCRperf,colorize = TRUE,print.cutoffs.at=seq(0,1,by=0.1),text.adj=c(-0.2,1.7))
#Picking Values between 0.1 and 0.2, we can conclude that we get the best least false positive rate(sensitivity) with p = 0.1
p = 0.1
class_predictions = ifelse(predictions_logistic > p,1,0)
conf_logistic_matrix = table(Test_data$Churn,class_predictions)
conf_logistic_matrix
#Sensitivity = 27.53%
198/(198+521)
#Specificity = 97.25%
922/(922+26)
#Accuracy = 67.18%
(922+198)/1667
#Implementing Xgboost For this classification Problem
#Preparing the data for Xgboost
#Reloading the Data To Prepare It For Xgboost Algorithm
Train_data <- data.frame(read_csv("E:/Customer_Churn_Edwisor_Project/Train_data.csv"))
Test_data <- data.frame(read_csv("E:/Customer_Churn_Edwisor_Project/Test_data.csv"))
Train_data$phone.number = NULL
Test_data$phone.number = NULL
Train_data[,list_removed] = NULL
Test_data[,list_removed] = NULL
Train_data$international.plan <- revalue(Train_data$international.plan, c("yes"= 1, "no"= 0))
Train_data$voice.mail.plan <- revalue(Train_data$voice.mail.plan, c("yes"= 1, "no"= 0))
Test_data$voice.mail.plan <- revalue(Test_data$voice.mail.plan, c("yes"= 1, "no"= 0))
Test_data$international.plan <- revalue(Test_data$international.plan, c("yes"= 1, "no"= 0))
Test_data$Churn <- revalue(Test_data$Churn, c("True."= 1, "False."= 0))
Train_data$Churn <- revalue(Train_data$Churn, c("True."= 1, "False."= 0))
Train_data$state = as.factor(Train_data$state)
Train_data$state = unclass(Train_data$state)
Test_data$state = as.factor(Test_data$state)
Test_data$state = unclass(Test_data$state)
Train_data[,2:16] = sapply(Train_data[,2:16],as.numeric)
Test_data[,2:16] = sapply(Test_data[,2:16],as.numeric)
#Fitting the model on the data
xgb <- xgboost(data = data.matrix(Train_data[,-1]),label = Train_data[,16],eta = 0.1,max_depth = 15,nrounds = 25)
#Making Predictions
y_pred <- predict(xgb, data.matrix(Test_data[,-1]))
#Turning Probabilities to class predictions by rounding the values
predictions_xg <- as.numeric(y_pred > 0.5)
#Finding the confusion matrix
conf_Xg_Matrix = table(Test_data$Churn,predictions_xg)
conf_Xg_Matrix
#Conclusion
#Tabulating the Results to decide on the best model










