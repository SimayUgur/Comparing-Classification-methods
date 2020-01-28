#install.packages("ISLR")
library(ISLR)
#install.packages("tidyverse")
library(tidyverse)
#install.packages("funModeling")
library(funModeling)
#install.packages("caret")
library(caret)
#install.packages("pROC")
library(pROC)
#install.packages("class")
#install.packages("ROCR")
library(ROCR) #roc icin
#install.packages("GGally")
library(GGally)
ges("rpart")
library(rpart)
#install.packages("cli")
library(cli)
#install.packages("tree")
library(tree)
#install.packages("rpart.plot")
library(rpart.plot)
#install.packages("randomForest")
library(randomForest)
#install.packages("DiagrammeR")
library(DiagrammeR)
#install.packages("mlbench")
library(mlbench)

library(mvtnorm)
library(caret)

sigma <- matrix(nrow = 8, ncol = 8, 0)
diag(sigma) = 1
mvn <- rmvnorm(n=500, mean=rnorm(8), sigma=sigma)
e=rnorm(500)
Y <- rowSums(mvn)+e

rmvm <- data.frame(cbind(Y, mvn))
hist(rmvm$Y)

names(rmvm) <- c("Y", paste("X", 1:8, sep = ""))



rmvm$Y[Y>ceiling(mean(Y))]=1
rmvm$Y[Y<=ceiling(mean(Y))]=0
rmvm$Y <- as.factor(rmvm$Y)

View(rmvm)

#verinin train-test diye ayrýlmasý
set.seed(123)
trainIndex <- createDataPartition(rmvm$Y, p = .7,list = FALSE, times = 1)
train <- rmvm[ trainIndex,]
test  <- rmvm[-trainIndex,]
str(train)
dim(train)
dim(test)
train_x <- train %>% dplyr::select(-Y)
train_y <- train$Y
test_x <- test %>% dplyr::select(-Y)
test_y <- test$Y


#tek bir veri seti
training <- data.frame(train_x, Sales = train_y)

#KARAR AÐAÇLARI

#Model
set.seed(1)
seat_tree <- tree(Y~., train)
summary(seat_tree)
#Agacin Gorsellestirilmesi
plot(seat_tree)
text(seat_tree, pretty = 0)

#Rpart fonksiyonu 
#görselleþtirme daha zengindir.
seat_rpart <- rpart(Y ~ ., data = train, method = "class")
summary(seat_rpart)
plotcp(seat_rpart)

#En iyi CP deðeri
min_cp <- seat_rpart$cptable[which.min(seat_rpart$cptable[,"xerror"]), "CP"]
min_cp

seat_rpart_prune <- prune(seat_rpart, cp = min_cp)
seat_rpart_prune
prp(seat_rpart_prune)

rpart.plot(seat_rpart_prune)
summary(seat_rpart_prune)

predict(seat_tree, train_x, type = "class")

predict(seat_tree, train_x, type = "vector")

tb <- table(predict(seat_tree, test_x, type = "class"), test_y)

confusionMatrix(tb, positive = "1")

#model Tuning

seat_tree_cv <- cv.tree(seat_tree, FUN = prune.misclass, K = 10)
min_tree <- which.min(seat_tree_cv$dev)
min_tree

seat_tree_cv$size[min_tree]

# Gorsel Incelenmesi
par(mfrow = c(1,2))
plot(seat_tree_cv)
plot(seat_tree_cv$size, 
     seat_tree_cv$dev / nrow(train), 
     type = "b",
     xlab = "Agac Boyutu/Dugum Sayisi", ylab = "CV Yanlis Siniflandirma Orani")

dev.off()

seat_tree_prune <- prune.misclass(seat_tree, best = 15)
summary(seat_tree_prune)
#Modellerin KArþýlaþtýrýlmasý

plot(seat_tree_prune)
text(seat_tree_prune, pretty = 0)

tb <- table(predict(seat_tree_prune, test_x, type = "class"), test_y)
confusionMatrix(tb, positive = "1")


#BAGGING
library(ipred) #bagging icin 
library(randomForest)
library(adabag)
library(ipred)
set.seed(1)
#1.yöntem
bag_fit1 <- ipredbagg(train_y, train_x)
bag_fit1


bag_fit2 <- bagging(Y~ ., data = train,coob=TRUE)
names(bag_fit2)
bag_fit2$trees

bag_fit <- randomForest(train$Y ~ . , data = train,
                        mtry = ncol(train) - 1,
                        importance = TRUE,
                        ntrees = 500
)
importance(bag_fit)
varImpPlot(bag_fit)


#Tahmin
pred <- predict(object = bag_fit, 
                newdata = test,  
                type = "class") 
print(pred)

pred<-predict(bag_fit, test_x,type="class")


defaultSummary(data.frame(obs = test_y,
                          pred = predict(bag_fit, test_x)))


plot(bag_fit, col = "dodgerblue", 
     lwd = 2, 
     main = "Bagged Trees: Hata ve Agac Sayisi Karsilastirimasi")
grid()


## Model Tuning

ctrl <- trainControl(method = "cv", number = 10)

mtry <- ncol(train_x)
tune_grid <- expand.grid(mtry = mtry) #arama iþleminde kullanýlacak olan deðer.



bag_tune <- train(train_x, train_y, 
                  method = "rf", 
                  tuneGrid = tune_grid,
                  trControl = ctrl)



defaultSummary(data.frame(obs = test_y,
                          pred = predict(bag_tune, test_x)))

#RANDOM FOREST
#model
set.seed(1)
rf_fit <- randomForest(train_x, train_y, importance = TRUE)

importance(rf_fit,scale=FALSE)

varImpPlot(rf_fit)


#Tahmin

predict(rf_fit, test_x)

confusionMatrix(predict(rf_fit, test_x), test_y, positive = "1")

#MODEL TUNING
##RANDOM SEARCH
control <- trainControl(method='repeatedcv', 
                        number = 10,
                        search = 'random')

#tunelenght ile 8 tane mtry degeri rastgele uretilecek 
set.seed(1)
rf_random <- train(Y ~ .,
                   data = train,
                   method = 'rf',
                   metric = 'Accuracy',
                   tuneLength  = 8, 
                   trControl = control)

plot(rf_random)
dev.off()

#GRID SEARCH
control <- trainControl(method='cv', 
                        number=10, 
                        search='grid')

tunegrid <- expand.grid(mtry = (1:8)) 

rf_gridsearch <- train(Y ~ ., 
                       data = train,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid=tunegrid)

plot(rf_gridsearch)
predict(rf_gridsearch, test_x)
confusionMatrix(predict(rf_gridsearch, test_x), test_y, positive = "1")

#logistic REGRESYON
MODEL:
model_glm <- glm(Y~., 
                 data = train, 
                 family = "binomial")
summary(model_glm)

levels(train$Y)[1]


summary(model_glm)
options(scipen = 9)
G<-485.95-157.95
1-pchisq(328,8)
#model Tune
library(MASS)
step.model<-model_glm %>% stepAIC(trace = FALSE)
G1<-485.9-158
1-pchisq(327,9-8)





  ## Tahmin


head(predict(step.model, type = "response"))

ol <- predict(step.model, type = "response")
summary(ol)
hist(ol)


model_glm_pred <- ifelse(predict(step.model, type = "response") > 0.5, "1","0")

table(step.model)

#Siniflandirma Hatasi Tespiti ve Karmasiklik Matrisi
class_err <- function(gercek, tahmin) {
        
        mean(gercek != tahmin)
        
}

#yanlis siniflandirma orani
class_err(train$Y, model_glm_pred)

1-class_err(train$Y, model_glm_pred)


tb <- table(tahmin = model_glm_pred, 
            gercek = train_y)

km <- confusionMatrix(tb, positive = "1")

c(km$overall["Accuracy"], km$byClass["Sensitivity"])

predictTEST<-predict(step.model,type="response",newdata=test_x)
tb1<- table(tahmin = predictTEST, 
            gercek = test_y)
#2)





#3)

## ROC Egrisi
dev.off()
model_glm <- glm(Y~ ., 
                 data = train, 
                 family = "binomial")


test_ol <- predict(model_glm, newdata = test_x, type = "response")

a <- roc(test_y ~ test_ol, plot = TRUE, print.auc = TRUE)
a$auc

#LDA -QDA

library("psych")

#Dogrusal ayristirma analizi
library(MASS)

model_lda<-lda(Y~.,data=train)
model_lda
attributes(model_lda) #neleri gorebilecegimizi gosterir

#histograms
tahmin_1<-predict(model_lda,train)
hist_lda1<-ldahist(data=tahmin_1$Y,g=train$Y)
hist_lda2<-ldahist(data=tahmin_1,g=train$Y)

#Bi-plot

library("devtools")
install_github("fawda123/ggord") 
library(ggord) 
ggord(model_lda,train$Y)

#Partition plots
library(klaR)
partimat(Y~., data=train, method="lda")

#Confusion matrix-accuracy-train
tahmin_1<-predict(model_lda,train)
cfmatrix_1<-table(Tahmin=tahmin_1$class, Gercek=train$Y)
cfmatrix_1
accuracy_1<-sum(diag(cfmatrix_1))/sum(cfmatrix_1)
#veya
accuracy_1<-mean(tahmin_1$class==train$Y)

#Confusion matrix-accuracy-test
tahmin_2<-predict(model_lda,test)
cfmatrix_2<-table(Tahmin=tahmin_2$class, Gercek=test$Y)
cfmatrix_2
accuracy_2<-sum(diag(cfmatrix_2))/sum(cfmatrix_2)
#veya
accuracy_2<-mean(tahmin_2$class==test$Y)


#QDA
model_qda<-qda(Y~.,data=train)
tahmin_qda_1<-predict(model_qda,train)
cfmatrix_qda_1<-table(Tahmin=tahmin_qda_1$class, Gercek=train$Y)
accuracy_qda_1<-mean(tahmin_qda_1$class==train$Y)

tahmin_qda_2<-predict(model_qda,test)
cfmatrix_qda_2<-table(Tahmin=tahmin_qda_2$class, Gercek=test$Y)
accuracy_qda_2<-mean(tahmin_qda_2$class==test$Y)
dev.off()

#Partition plots
library(klaR)
partimat(Y~., data=train, method="qda")
 #soru2
sigma <- matrix(nrow = 8, ncol = 8, 0)
diag(sigma) = 1
mvn <- rmvnorm(n=500, mean=rnorm(8), sigma=sigma)
e=rnorm(500)
Y <- rowSums(mvn)+e

rmvm <- data.frame(cbind(Y, mvn))
hist(rmvm$Y)

names(rmvm) <- c("Y", paste("X", 1:8, sep = ""))



rmvm$Y[Y>ceiling(mean(Y))]=1
rmvm$Y[Y<=ceiling(mean(Y))]=0
rmvm$Y <- as.factor(rmvm$Y)

View(rmvm)

#verinin train-test diye ayrýlmasý
set.seed(123)
trainIndex <- createDataPartition(rmvm$Y, p = .7,list = FALSE, times = 1)
train <- rmvm[ trainIndex,]
test  <- rmvm[-trainIndex,]
str(train)
dim(train)
dim(test)
train_x <- train %>% dplyr::select(-Y)
train_y <- train$Y
test_x <- test %>% dplyr::select(-Y)
test_y <- test$Y


#tek bir veri seti
training <- data.frame(train_x, Sales = train_y)



mod_rf <- train(Y ~ .,
                data=train, 
                method='rf', 
                trControl=trainControl(method="cv", 
                                       number=4, 
                                       allowParallel=TRUE, 
                                       verboseIter=TRUE)) # Random Forrest
mod_tree <- train(Y ~ ., 
                  data=train, 
                  method='rpart',
                  trControl=trainControl(method="cv", 
                                         number=4, 
                                         allowParallel=TRUE, 
                                         verboseIter=TRUE)) # Trees
mod_glm <- train(Y ~ ., 
                 data=train, 
                 method='glm', 
                 trControl=trainControl(method="cv", 
                                        number=4, 
                                        allowParallel=TRUE, 
                                        verboseIter=FALSE)) # logistic regresyon
mod_lda <- train(Y ~ ., 
                 data=train, 
                 method='lda',
                 trControl=trainControl(method="cv", 
                                        number=4, 
                                        allowParallel=TRUE, 
                                        verboseIter=TRUE)) # Linear Discriminant Analysis
mod_nb <- train(Y ~ ., 
                data=train, 
                method='qda',
                trControl=trainControl(method="cv", 
                                       number=4, 
                                       allowParallel=TRUE, 
                                       verboseIter=TRUE))# QDA



#Predictions Against Cross Validation Data
##For each candidate model, predictions are made agaist the cross-validation data set.
##Then, a confusion matrix is calculated and stored for each model for later reference.
pred_rf <- predict(mod_rf,test)
cm_rf <- confusionMatrix(pred_rf,test$Y)

pred_tree <- predict(mod_tree,test)
cm_tree <- confusionMatrix(pred_tree,test$Y)

pred_glm <- predict(mod_glm,test)
cm_gbm <- confusionMatrix(pred_glm,test$Y)

pred_lda <- predict(mod_lda,test)
cm_lda <- confusionMatrix(pred_lda,test$Y)

pred_qda <- predict(mod_nb,test)
cm_qda <- confusionMatrix(pred_qda,test$Y)


#The accuracy results are accessible within the confusion matrix object derived from each model.
#To easily analyze them, the out of sample accuracy is aggregated
#for each model and plotted on a common index.



model_compare <- data.frame(Model = c('Random Forest',
                                      'Trees',
                                      'LOJISTIC REGRESYON',
                                      'Linear Discriminant',
                                      'QDA'),
                            Accuracy = c(cm_rf$overall[1],
                                         cm_tree$overall[1],
                                         cm_gbm$overall[1],
                                         cm_lda$overall[1],
                                         cm_qda$overall[1]))

ggplot(aes(x=Model, y=Accuracy), data=model_compare) +
  geom_bar(stat='identity', fill = 'blue') +
  ggtitle('Cross-Validation Veride Modelerin Doðruluklarýnýn karþýlaþtýrýlmasý') +
  xlab('Modeller') +
  ylab('Overall Accuracy')


ggplot(data=data.frame(model_compare$table)) + 
  geom_tile(aes(x=Reference, y=Prediction, fill=Freq)) +
  ggtitle('Prediction Accuracy for Classes in Cross-Validation (Decision Tree Model)') +
  xlab('Actual Classe') +
  ylab('Predicted Classe from Model')






dev.off()
# Calculate sensitivity and false positive measure for random forest mode# plot ROC curves
# egrisi (iki grup icin)
# Generate the test set AUCs using the two sets of predictions & compare



library(pROC)
data(ROCR.simple)
library(ROCR) 


library(ROCR)




#3)
library(Metrics)

actual <- test$Y
rf_auc <- auc(actual = actual, predicted =pred_rf  )
tr_auc <- auc(actual = actual, predicted =pred_tree)
glm_auc <- auc(actual = actual, predicted = pred_glm)
lda_auc <- auc(actual = actual, predicted = pred_lda)
qda_auc <- auc(actual = actual, predicted = pred_qda)

# Print results
print(paste("Random Forest Test AUC: %.3f", rf_auc))
print("Trees Test AUC: %.3f",tr_auc )
print("LR: %.3f", glm_auc)
print("LDA Test AUC: %.3f", lda_auc)
print("QDA Test AUC: %.3f", qda_auc)


preds <- c(pred_rf, pred_tree, pred_glm,pred_lda,pred_qda)

# List of actual values (same for all)
m <- length(preds)
actuals <- rep(list(test$Y), m)
length(actuals)
length(preds)
act<-as.vector(actuals)

# Plot the ROC curves
if (!is.numeric( unlist( predictions ))) {
  stop("Currently, only continuous predictions are supported by ROCR.")
}
pred <- prediction(preds, actuals,label.ordering = NULL)
rocs <- performance(pred , "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Decision Tree", "Bagged Trees", "Random Forest", "GBM"),
       fill = 1:m)



###
# Fit the model
model_ct <- rpart(Y~., data = train, method = "class")

# plot the model
# install.packages("RColorBrewer") #
library(RColorBrewer)
pal <- brewer.pal(n=10, name = "YlGn")
rpart.plot(model_ct, yesno = 2, type = 2, extra = 0,fallen.leaves = F, shadow.col = "darkgray", box.palette = pal)

# plot the cp table
plotcp(model_ct)
print(model_ct$cptable)

# Retrieve optimal cp value based on cross-validated error(xerror)
opt_index <- which.min(model_ct$cptable[, "xerror"])
cp_opt <- model_ct$cptable[opt_index, "CP"]

# Prune the model (to optimized cp value)
model_ct_opt <- prune(tree = model_ct,
                      cp = cp_opt)

# Plot the optimized model
rpart.plot(x = model_ct_opt, yesno = 2,type = 2, extra = 0, fallen.leaves = F, shadow.col = "darkgray", box.palette = pal)

# predict using optimized tree
proba_ct <- predict(object = model_ct_opt, newdata = test)
ct_pr <- predict(model_ct_opt, newdata = test, type = "class")
ct_pr
# Define a function to calculate accuracy
accuracy <- function(pred){
  pred <- as.data.frame(pred)
  pred$class <- as.numeric(pred$`1` > pred$`0`)
  result <- mean(test$Y == pred$class)
  print(result)
  return(result)
}

# Calculate accuracy
acc_ct <- accuracy(proba_ct)

##### Bagging #####

# Fit the model
## install.packages("ipred")
library(ipred)
ct_bag <- bagging(formula = Y~., data = train, coob = T)

# predict using bagged tree
proba_bag <- predict(object = ct_bag, newdata = test, type = "prob")
acc_bag <- accuracy(proba_bag)

##### Random Forest #####
## install.packages("randomForest")
library(randomForest)
rf <- randomForest(Y~., data=train)
proba_rf <-  predict(object = rf, newdata = test, type = "prob")
acc_rf <- accuracy(proba_rf)


##### Logistic Regression #####
# fit the model
logreg <- glm(formula = Y~., family = binomial, data = train)
summary(logreg)

# predict using logistic regression
proba_lr <- predict(object = logreg, newdata = test, type = "response")
pred_logreg <- ifelse(test = prob_lr >= mean(pred_lr), yes = 1, no = 0)

# calculate accuracy
acc_logreg <- mean(pred_logreg == test$Y)


##### Linear Discriminant Analysis #####
library(MASS)
# fit the model
model_lda <- lda(Y~., data = train)

# predict using lda, getting both classes and probabilities
pred_lda <- predict(object = model_lda, newdata = test)
prob_lda <- pred_lda$posterior
proba_lda <- ifelse(test = (prob_lda[,1] > prob_lda[,2]), yes = prob_lda[,1], no= prob_lda[,2])
proba_lda

# calculate accuracy using lda
acc_lda <- mean(pred_model_lda$class == test$Y)


##### Quadratic Discriminany Analysis #####
model_qda<-qda(Y~.,data=train)

# predict using qda, getting both classes and probabilities
pred_qda <- predict(object = model_qda, newdata = test)
prob_qda <- pred_qda$posterior
proba_qda <- ifelse(test = (prob_qda[,1] > prob_qda[,2]), yes = prob_qda[,1], no= prob_qda[,2])
proba_qda

# calculate accuracy using qda
acc_qda <- mean(pred_qda$class == test$Y)


########## ROC - AUC ########
library(pROC) 

# getting 1's probabilities
probab_ct <- proba_ct[,2]
probab_bag <- proba_bag[,2]
probab_rf <- proba_rf[,2]
# roc of ct
roc_ct <- roc(response = test$Y, predictor = probab_ct, plot = T, 
              legacy.axes = T, percent = T,
              main = "ROC Curve of CT",
              xlab = "False Positive Percentage",
              ylab = "True Positive Percentage",
              col = "#41ae76", lwd = 2, print.auc = T, print.auc.y = 10)


roc_bag <- roc(response = test$Y, predictor = probab_bag, plot = T, 
               legacy.axes = T, percent = T,
               main = "ROC Curve of Bagging",
               xlab = "False Positive Percentage",
               ylab = "True Positive Percentage",
               col = "#02818a", lwd = 2, print.auc = T, print.auc.y = 20, add = T)

roc_rf <- roc(response = test$Y, predictor = probab_rf, plot = T, 
              legacy.axes = T, percent = T,
              main = "ROC Curve of RF",
              xlab = "False Positive Percentage",
              ylab = "True Positive Percentage",
              col = "#005824", lwd = 2, print.auc.y = 30,print.auc = T, add = T)

roc_lr <- roc(response = test$Y, predictor = proba_lr, plot = T, 
              legacy.axes = T, percent = T,
              main = "ROC Curve of LogReg",
              xlab = "False Positive Percentage",
              ylab = "True Positive Percentage",
              col = "#df65b0", lwd = 2, print.auc = T, print.auc.y = 40, add = T)


roc_lda <- roc(response = test$Y, predictor = proba_lda, plot = T, 
               legacy.axes = T, percent = T,
               main = "ROC Curve of LDA",
               xlab = "False Positive Percentage",
               ylab = "True Positive Percentage",
               col = "#807dba", lwd = 2, print.auc = T, print.auc.y = 50, add = T)

roc_qda <- roc(response = test$Y, predictor = proba_qda, plot = T, 
               legacy.axes = T, percent = T,
               main = "ROC Curve of QDA",
               xlab = "False Positive Percentage",
               ylab = "True Positive Percentage",
               col = "#4a1486", lwd = 2, print.auc = T, print.auc.y = 60, add = T)
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Decision Tree", "Bagged Trees", "Random Forest", "LR","LDA","QDA"),
       fill = c("#41ae76","#02818a","#005824","#df65b0","#807dba","#4a1486"))




