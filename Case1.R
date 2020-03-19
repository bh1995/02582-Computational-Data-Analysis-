
## 02582 COMPUTATIONAL DATA ANALYSIS ##

setwd("C:/Users/Bjorn/OneDrive/Dokument/University/DTU/02582 Computational Data Analysis/case1/Case Data")

# Load needed packages
library(dplyr) # dplyr is package used to manipulate data
library(tidyr)
library(ggplot2)
library(glmnet)
library(caret)
library(neuralnet)
library(kernlab)
library(pracma) # for function logspace
library(pander)
library(tree)
library(randomForest)
library(mboost)



data = read.csv("case1Data.txt", sep=",", header=TRUE, na.strings="NA")
n = dim(data)[1] # number of observations
p = dim(data)[2] # number of parameters

# function to find the most common level (value) for a given column
calculate_mode = function(x) { # x is the data vector 
  uniqx = unique(na.omit(x))
  uniqx[which.max(tabulate(match(x, uniqx)))]
}

mcl_97 = calculate_mode(data[,97]) # most common level for column 97 is k
mcl_98 = calculate_mode(data[,98]) # most common level for column 98 is J
mcl_99 = calculate_mode(data[,99]) # most common level for column 99 is H
mcl_100 = calculate_mode(data[,100]) # most common level for column 100 is H
mcl_101 = calculate_mode(data[,101]) # most common level for column 101 is H

# replacing all NaN values with the columns most common level
data$C_.1 = as.character(data$C_.1)
data$C_.1[data$C_.1 == " NaN"] = "K"
data$C_.1 = as.factor(data$C_.1)

data$C_.2 = as.character(data$C_.2)
data$C_.2[data$C_.2 == " NaN"] = "J"

data$C_.3 = as.character(data$C_.3)
data$C_.3[data$C_.3 == " NaN"] = "H"

data$C_.4 = as.character(data$C_.4)
data$C_.4[data$C_.4 == " NaN"] = "H"

data$C_.5 = as.character(data$C_.5)
data$C_.5[data$C_.5 == " NaN"] = "H"

# Encode factor variables by converting them to multiple "dummy variables" (aka "one hot" encoding)
dmy = dummyVars(" ~ . ", data=data)
data2 = data.frame(predict(dmy, newdata=data))

# Normalize data
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Scale/ normalize data
maxs = apply(data2, 2, max) 
mins = apply(data2, 2, min)
scaled = as.data.frame(scale(data2, center = mins, scale = maxs - mins))

normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
scaled2 = data.frame(y=data2[,1], scale(data2[,2:126]))

# Split scaled data into train and test (80/20)
n=dim(scaled2)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.8))
train_=scaled[id,]
test_=scaled[-id,]

# Split unscaled data into train and test (80/20)
n=dim(data2)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.8))
train=data2[id,]
test=data2[-id,]

# Function to quickly calculate RMSE 
rmse = function(error){
  sqrt(mean(error^2))
}

# Apply PCA
prc = prcomp(data2, scale=TRUE)
plot(prc)
summary(prc)
print(prc)

# Apply BH test to parameters
pvals = sapply(2:126, function(i)
  t.test(data2[,i]~data2$y, alternative="two.sided")$p.value)
names(pvals)<-colnames(data2[,-1])
# pvals[which(pvals<=0.05)]
sorted_pvals<-sort(pvals,decreasing =FALSE)
# L=sapply(1:4702, function(i) max(pvals[i],0.05*(i/4702)))
adjusted<-p.adjust(pvals,method="BH")
print(data.frame(features=names(pvals)[which(adjusted<0.05)]))

# fit a elastic net regression model
model_cvelastic1 = cv.glmnet(y=as.matrix(train[,1]), x=as.matrix(train[,-1]), scale=TRUE, alpha=0.2) # alpha=0 gives ridge regression
plot(model_cvelastic1) # Check to make sure cross-validation looks ok
s = summary(model_cvelastic1)
model_elastic1 = glmnet(y=as.matrix(train[,1]), x=as.matrix(train[,2:126]), lambda=0.05)

y_hat = predict(model_cvelastic1, newx=as.matrix(test[,2:126]), s=model_cvelastic1$lambda.1se) # lambda.1se = leave one out method
rmse.glm0 = rmse(y_hat - test$y)
rmse.glm0

plot(test$y, y_hat,col='red',main='Real vs predicted elastic', pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='Elastic Net',pch=18,col='red', bty='n')

# Alternative ridge regression
n = dim(data2)[1] #nr. observations
#p = dim(data2)[2] #nr. parameters to estimate
p = 125
# y = data[,9]
# x = data[,1:8]
x = as.matrix(data2[,-1])
y = as.matrix(data2[,1])
xn = scale(x)
yn = y-mean(y)

m = 100; # try m values of lambda
lambdas = logspace(-3,4,m); # define k values of lambda on a log scale
betas_r = matrix(NaN,p,m); # prepare a matrix for all the ridge parameters
for (i in 1:m){ # repeat m times
  betas_r[,i] = solve(t(xn)%*%xn + lambdas[i]*diag(p), t(xn)%*%yn) # estimte ridge coefficients #ginv()?
}

# plot the results (log x-axis, linear y-axis), exclude the intercept
rainbowvec = rainbow(p-1)
plot(lambdas, betas_r[2,], log='x',type='l',ylim=c(-1/2,1),xlab=expression(lambda),
     ylab="Estimate",col=rainbowvec[1]) # ignore log warning, it plots in log scale just fine
for (i in 3:p){
  lines(lambdas, betas_r[i,],col=rainbowvec[i])
}


# fit lm regression model
model_lm = lm(y~., data=train)
y_hatlm = predict(model_lm, newdata=test)
rmse.lm = rmse(y_hatlm - test$y)

# fit nn
# 2 hidden layers with configuration: 13:5:3:1. The input layer has 13 inputs,
# the two hidden layers have 5 and 3 neurons and the output layer has, of course,
# a single output since we are doing regression.
nms = names(train_)
f = as.formula(paste("y ~", paste(nms[!nms %in% "y"], collapse = " + ")))
model_nn = neuralnet(f, data=train_, hidden=c(5,3), linear.output=TRUE)

y_hat.nn = compute(model_nn, test_[,2:126])

y_hat.nn_ = y_hat.nn$net.result*(max(data2$y)-min(data2$y))+min(data2$y) # This will scale the predictions back to original
test.r = (test_$y)*(max(data2$y)-min(data2$y))+min(data2$y) # Scale test data back to original
rmse.nn = rmse(y_hat.nn_ - test.r)
rmse.nn

# fit SVM
model_svm = ksvm(x=as.matrix(train[,2:126]), y=as.matrix(train[,1]), scaled=T, C=5) # Cost constraint of 5
y_hat.svm = predict(model_svm, test[2:126])
#y_hat.svm_ = y_hat.svm*(max(data2$y)-min(data2$y))+min(data2$y)
rmse.svm = rmse(y_hat.svm - test[,1])
rmse.svm
# Calculate rmse for SVM using training data
y_hat.svm_train = predict(model_svm, train[2:126])
rmse.svm_train = rmse(y_hat.svm_train - train[,1])
rmse.svm_train


# Fit random forrest
model_rf = randomForest(y~., data=train, ntree=25)
  #randomForest(x=train[,-1], y=train[,1], scaled=T)
plot(model_rf)
y_hat.rf = predict(model_rf, newdata=test[,-1])
rmse.rf = rmse(y_hat.rf - test[,1])
rmse.rf
test.err = double(125)
for(i in 2:126){
  model_rf = randomForest(y~., data=train, mtry=i)
  #randomForest(x=train[,-1], y=train[,1], scaled=T)
  y_hat.rf = predict(model_rf, newdata=test[,-1])
  test.err[i] = rmse(y_hat.rf - test[,1])
}
test.err
plot(test.err)
# Calculate rmse for random forest using training data
y_hat.rf_train = predict(model_rf, train[,2:126])
rmse.rf_train = rmse(y_hat.rf_train - train[,1])
rmse.rf_train

# Find importance of parameters with random forrest model, higher -> more important
round(importance(model_rf), 2)

# Fit regression tree

# Fit Adaboost model
model_ada = blackboost(formula=y~., data=train, control=boost_control(mstop=100), family=GaussReg())
y_hat.ada = predict(model_ada, newdata=test)
rmse.ada = rmse(y_hat.ada - test[,1])
rmse.ada

# Fit gbm (gradiaent boosting model)
library(gbm)
model_gbm = gbm(y~., data=train)
y_hat.gbm = predict(model_gbm, newdata=test, n.trees=100)
rmse.gbm = rmse(y_hat.gbm - test[,1])
rmse.gbm # Seems to be best model so far

# Regression tree with library rpart
library(e1071)
library(caret)
library(rpart)
numFolds <- trainControl(method = "cv", number = 10) # Produce CV train sets (k = 10)
cpGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01)) # To try many different cp values
model_rpart1 <- train(y~., data = train, method = "rpart", trControl = numFolds, tuneGrid = cpGrid) # lowest rmse found with cp=0.1
model_rpart2 = rpart(y~., data = train, method = "anova", cp = 0.1) 
y_hat.rpart <- predict(model_rpart2, newdata = test)
rmse.rpart = rmse(y_hat.rpart - test[,1])
rmse.rpart # Still higher than randomForest package used above ...
