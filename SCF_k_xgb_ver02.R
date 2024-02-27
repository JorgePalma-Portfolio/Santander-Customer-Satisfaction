# 0.832204 499 - 0.8430626 - 0.832669 - max_depth=12
# 0.83704  422 - 0.8468958 - 0.836528 - max_depth=10
# 0.838534 500 - 0.8480673 - 0.838309 - max_depth=5
# 0.844881 595 - 0.8109322 - 0.839409 - max_depth=5 - k=2
# 0.845243 918 - 0.8095308 - 0.840368 - max_depth=5 - k=2

# 0.845484 899
# 0.845516 956 
# 0.845713 902
# 0.845715 956

library(Metrics)
library(xgboost)
library(caret)
library(cvTools)

train <- read.csv("train/train.csv")
test  <- read.csv("test/test.csv")

# Identify train and test data
Train_Flag <- rep(1,nrow(train))
train <- cbind(train[,c(1,371)],Train_Flag,train[-c(1,371)])

Train_Flag  <- rep(0,nrow(test))
test <- cbind(ID=test[,1],TARGET=NA,Train_Flag,test[,-c(1)])
rm(Train_Flag)

# Combine train and test datasets, name all_data 
all_data <- rbind(train,test)

rm(test,train)

all_data <- data.frame(all_data[with(all_data,order(ID)),])

all_data[all_data == -999999] <- -1
all_data[all_data == 9999999999] <- -1

# fields.cluster <-  c("num_var5_0","num_var5","saldo_var5",
#                      "saldo_medio_var5_hace2","saldo_medio_var5_hace3",      
#                      "saldo_medio_var5_ult1","saldo_medio_var5_ult3","var15","saldo_var30")
# 
# data_cluster <- all_data[,fields.cluster]
# 
# pmatrix <- scale(data_cluster)
# 
# pclusters <- kmeans(pmatrix,20,nstart=50)
# 
# all_data <- data.frame(all_data,cluster=pclusters$cluster)
# 
# rm(pmatrix)

novariation.fields <- c("ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28", "ind_var27", "ind_var41", 
                        "ind_var46_0", "ind_var46", "num_var27_0", "num_var28_0", "num_var28", "num_var27", "num_var41", 
                        "num_var46_0", "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46", "imp_amort_var18_hace3",
                        "imp_amort_var34_hace3", "imp_reemb_var13_hace3", "imp_reemb_var33_hace3", "imp_trasp_var17_out_hace3", 
                        "imp_trasp_var33_out_hace3", "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3", "num_reemb_var33_hace3", 
                        "num_trasp_var17_out_hace3", "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3")

#all_data <- all_data[,!(names(all_data) %in% novariation.fields)]

data_1 <- list()
data_1$custom0 <- log1p(all_data$var15)
data_1$custom1 <- log1p(all_data$var38)
data_1$custom2 <- 1/exp(all_data$saldo_var30)
data_1$custom3 <- all_data$saldo_medio_var5_ult3/all_data$var15
data_1$custom4 <- all_data$saldo_var30/all_data$var15
data_1$custom5 <- all_data$var38/all_data$var15

data_1$custom6 <-  all_data$saldo_medio_var5_hace3 + all_data$saldo_medio_var5_hace2

data_1$custom7 <- 1/exp(all_data$saldo_medio_var5_hace3)
data_1$custom8 <- 1/exp(all_data$num_var4)

all_data <- data.frame(all_data,data_1) 

fields.00 <- -c(1:3)

train.set <- all_data[all_data$Train_Flag==1,] 
test.set  <- all_data[all_data$Train_Flag==0,]  

k.fold <- 10

for(k in 3:3){
  
  set.seed(73)
  folds <- createFolds(train.set$TARGET,k.fold)
  
  ul.folds <- unlist(folds)
  
  train      <- ul.folds[!ul.folds %in% folds[[k]]] #Set the training set
  validation <- folds[[k]]                          #Set the validation set
  
  train_70 <- train.set[train,fields.00]
  train_30 <- train.set[validation,fields.00]
  y <- data.matrix(train.set[train,2])
  actual <- train.set[validation,2]
  
  xgtrain <- xgb.DMatrix(as.matrix(train_70), label = y,missing=-1)
  xgval <- xgb.DMatrix(as.matrix(train_30), missing=-1)
  watchlist <- list('train_70' = xgtrain)
  
  param0 <- list(
    "objective"  = "binary:logistic"
    , "eval_metric" = "auc"
    , "eta" = 0.009
    , "subsample" = 0.9
    , "colsample_bytree" = 0.5
    , "min_child_weight" = 1
    , "max_depth" = 5      # 6->   0.845298 656
    #    , "lambda" = 0.5
    #    , "gamma" = 0.01
  )
  
  model_cv = xgb.cv(
    params = param0
    , nrounds = 1500
    , nfold = 10
    , data = xgtrain
    , early.stop.round = 1000
    , maximize = FALSE
    , nthread = 6
    , print.every.n = 100
  )
  
  best <- max(model_cv$test.auc.mean)
  bestIter <- which(model_cv$test.auc.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  iter <- bestIter-1
  
  model <- list()
  nmodels <- 5
  spred <- rep(0,nrow(train_30))
  set.seed(73)
  for(i in 1:nmodels){
    
    model[[i]] = xgb.train(nrounds = iter, 
                           params = param0, 
                           data = xgtrain, 
                           watchlist = watchlist, 
                           print.every.n = 100,
                           nthread = 10,
                           verbose = 1)
    
    predicted <- predict(model[[i]], xgval)
    #cat(auc(actual, predicted),"\n")
    spred <- spred + predicted
  }
  spred <- spred/nmodels
  cat(k,auc(actual, spred),"<====== \n")
}  

xgsub <- xgb.DMatrix(data.matrix(test.set[,fields.00]))
spred <- rep(0,nrow(test.set))

for(i in 1:nmodels){
  predicted <- predict(model[[i]], xgsub)
  spred <- spred + predicted
}
spred <- spred/nmodels

submission <- read.table("test/sample_submission.csv", header=TRUE, sep=',')

submission$TARGET <- spred
write.csv(submission, "submission_xgb.csv", row.names=F, quote=F)

#-----#

names <- dimnames(train_70)[[2]]
importanceMatrix01 <- xgb.importance(names,model=model[[1]])
write.csv(importanceMatrix01,"impmatrix01.csv",row.names = F)

importanceMatrix02 <- xgb.importance(names,model=model[[2]])
write.csv(importanceMatrix02,"impmatrix02.csv",row.names = F)

importanceMatrix03 <- xgb.importance(names,model=model[[3]])
write.csv(importanceMatrix03,"impmatrix03.csv",row.names = F)






xgb.plot.importance(importanceMatrix[1:20,])

xgb.plot.tree(feature_names = names, model = model, n_first_tree = 2)

model.desc <- xgb.dump(model, with.stats = T)

