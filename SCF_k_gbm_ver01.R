# 0.8390225 

library(Metrics)
library(gbm)
library(caret)
library(cvTools)
library(digest)

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

all_data[all_data == -999999] <- 300
all_data[all_data == 9999999999] <- NA

all_data.s <- all_data[,-c(1,2,3)]
all_data.s <- all_data.s[!duplicated(lapply(all_data.s, digest))]
all_data.s <- all_data.s[,apply(all_data.s,2,sd) > 0]
all_data.s <- data.frame(scale(all_data.s))
pcomp      <- princomp(all_data.s)

data_0 <- data.frame(pc01=pcomp$scores[,1], pc02=pcomp$scores[,2],pc03=pcomp$scores[,3])

all_data <- cbind(all_data,data_0)

novariation.fields <- c("ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28", "ind_var27", "ind_var41", 
                        "ind_var46_0", "ind_var46", "num_var27_0", "num_var28_0", "num_var28", "num_var27", "num_var41", 
                        "num_var46_0", "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46", "imp_amort_var18_hace3",
                        "imp_amort_var34_hace3", "imp_reemb_var13_hace3", "imp_reemb_var33_hace3", "imp_trasp_var17_out_hace3", 
                        "imp_trasp_var33_out_hace3", "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3", "num_reemb_var33_hace3", 
                        "num_trasp_var17_out_hace3", "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3")

all_data <- all_data[,!(names(all_data) %in% novariation.fields)]

# data_1 <- list()
# data_1$custom0  <- log1p(all_data$var15)
# data_1$custom1  <- log1p(all_data$var38)
# data_1$custom2  <- 1/exp(-all_data$saldo_var30)
# data_1$custom3  <- all_data$saldo_medio_var5_ult3/all_data$var15
# 
# data_1 <- data.frame(data_1)

fields.00 <- -c(1:3)

train.set <- all_data[all_data$Train_Flag==1,] 
test.set  <- all_data[all_data$Train_Flag==0,]  

k.fold <- 10

set.seed(73)
folds <- createFolds(train.set$TARGET,k.fold)

ul.folds <- unlist(folds)

for(k in 1:k.fold){
  
  train      <- ul.folds[!ul.folds %in% folds[[k]]] #Set the training set
  validation <- folds[[k]]                          #Set the validation set
  
  train_70 <- data.frame(TARGET=train.set[train,2],train.set[train,fields.00])
  train_30 <- data.frame(TARGET=train.set[validation,2],train.set[validation,fields.00])
  
  interaction.depth <- floor(sqrt(ncol(train_70)))
  shrinkage <- 0.001
  
  GBM_train <- gbm(TARGET ~ .,
                   data=train_70,
                   n.trees=400,
                   distribution = "bernoulli",
                   interaction.depth=12,
                   n.minobsinnode=50, #40,
                   shrinkage=0.1,
                   #cv.folds=5,
                   #n.cores=4,
                   train.fraction=0.9,
                   bag.fraction=0.9,
                   verbose=T)
  
  saved_GBM <- GBM_train
  GBM_train$opt_tree <- gbm.perf(GBM_train, plot.it = F,  method="OOB") #Use the OOB method to determine the optimal number of trees
  summary(GBM_train,n.trees=GBM_train$opt_tree)
  
  predicted <- predict(GBM_train,train_30,GBM_train$opt_tree,type="response")
  
  actual <- train.set[validation,2]
  
  cat(auc(actual, predicted),"\n")
  
}

pretty.gbm.tree(saved_GBM, i.tree=1)


submission <- read.table("test/sample_submission.csv", header=TRUE, sep=',')

submission$TARGET <- spred
write.csv(submission, "submission_xgb.csv", row.names=F, quote=F)

