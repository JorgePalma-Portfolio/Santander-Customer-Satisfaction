# 0.832204 499 - 0.8430626 - 0.832669 - max_depth=12
# 0.83704  422 - 0.8468958 - 0.836528 - max_depth=10
# 0.838534 500 - 0.8480673 - 0.838309 - max_depth=5
# 0.844881 595 - 0.8109322 - 0.839409 - max_depth=5 - k=2
# 0.845243 918 - 0.8095308 - 0.840368 - max_depth=5 - k=2

# 0.845484 899
# 0.845516 956 
# 0.845713 902
# 0.845847 968 

library(Metrics)
library(xgboost)
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

all_data$numzero <- rowSums(all_data[,-c(1:3)] == 0)

rm(test,train)

all_data <- data.frame(all_data[with(all_data,order(ID)),])

all_data[all_data == -999999] <- 300

all_data.s <- all_data[,-c(1,2,3)]
all_data.s <- all_data.s[!duplicated(lapply(all_data.s, digest))]
all_data.s <- all_data.s[,apply(all_data.s,2,sd) > 0]
all_data.s <- data.frame(scale(all_data.s))
pcomp      <- princomp(all_data.s)

data_0 <- data.frame(pc01=pcomp$scores[,1], pc02=pcomp$scores[,2], pc03=pcomp$scores[,3], pc04=pcomp$scores[,4])

all_data <- cbind(all_data,data_0)

include.fields <- c("ID","TARGET","Train_Flag","num_var39_0", "ind_var13", "num_op_var41_comer_ult3","num_var43_recib_ult1","imp_op_var41_comer_ult3",  
                    "num_var8", "num_var42", "num_var30", "saldo_var8", "num_op_var39_efect_ult3", "num_op_var39_comer_ult3",  
                    "num_var41_0", "num_op_var39_ult3", "saldo_var13", "num_var30_0", "ind_var37_cte", "ind_var39_0",  
                    "num_var5", "ind_var10_ult1", "num_op_var39_hace2", "num_var22_hace2", "num_var35", "ind_var30", "num_med_var22_ult3",  
                    "imp_op_var41_efect_ult1", "var36", "num_med_var45_ult3", "imp_op_var39_ult1", "imp_op_var39_comer_ult3",  
                    "imp_trans_var37_ult1", "num_var5_0", "num_var45_ult1", "ind_var41_0", "imp_op_var41_ult1", "num_var8_0", 
                    "imp_op_var41_efect_ult3", "num_op_var41_ult3", "num_var22_hace3", "num_var4", "imp_op_var39_comer_ult1", 
                    "num_var45_ult3", "ind_var5", "imp_op_var39_efect_ult3", "num_meses_var5_ult3", "saldo_var42",  
                    "imp_op_var39_efect_ult1", "pc01", "num_var45_hace2", "num_var22_ult1", "saldo_medio_var5_ult1", "pc02",
                    "saldo_var5", "ind_var8_0", "ind_var5_0", "num_meses_var39_vig_ult3", "saldo_medio_var5_ult3",  
                    "num_var45_hace3", "num_var22_ult3", "saldo_medio_var5_hace3", "saldo_medio_var5_hace2", "numzero",  
                    "saldo_var30", "var38", "var15")


novariation.fields <- c("ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28", "ind_var27", "ind_var41", 
                        "ind_var46_0", "ind_var46", "num_var27_0", "num_var28_0", "num_var28", "num_var27", "num_var41", 
                        "num_var46_0", "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46", "imp_amort_var18_hace3",
                        "imp_amort_var34_hace3", "imp_reemb_var13_hace3", "imp_reemb_var33_hace3", "imp_trasp_var17_out_hace3", 
                        "imp_trasp_var33_out_hace3", "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3", "num_reemb_var33_hace3", 
                        "num_trasp_var17_out_hace3", "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3")

all_data <- all_data[,!(names(all_data) %in% novariation.fields)]

data_1 <- list()
data_1$custom0 <- log1p(all_data$var15)
data_1$custom1 <- log1p(all_data$var38)
data_1$custom2 <- 1/exp(all_data$saldo_var30)
data_1$custom3 <- all_data$saldo_medio_var5_ult3/all_data$var15
data_1$custom4 <- all_data$saldo_var30/all_data$var15

all_data <- data.frame(all_data,data_1) 

fields.00 <- -c(1:3)

train.set <- all_data[all_data$Train_Flag==1,] 
test.set  <- all_data[all_data$Train_Flag==0,]  

k.fold <- 10

spred <- rep(0,7602)
model <- list()
predicted <- list()
i <- 1

for(k in c(2,10)){
  
  set.seed(73)
  folds <- createFolds(train.set$TARGET,k.fold)
  
  ul.folds <- unlist(folds)
  
  train      <- ul.folds[!ul.folds %in% folds[[k]]] #Set the training set
  validation <- folds[[k]]                          #Set the validation set
  
  train_70 <- train.set[train,fields.00]
  train_30 <- train.set[validation,fields.00]
  y <- data.matrix(train.set[train,2])
  actual <- train.set[validation,2]
  
  xgtrain <- xgb.DMatrix(as.matrix(train_70), label = y,missing=9999999999)
  xgval <- xgb.DMatrix(as.matrix(train_30), missing=9999999999)
  watchlist <- list('train_70' = xgtrain)
  
  param0 <- list(
    "objective"  = "binary:logistic"
    , "eval_metric" = "auc"
    , "eta" = 0.02 #0.03 #0.009
    , "subsample" = 0.8 #0.9
    , "colsample_bytree" = 0.7 #0.5
    , "min_child_weight" = 1
    , "max_depth" = 5      # 6->   0.845298 656
    #    , "lambda" = 0.5
    #    , "gamma" = 0.01
  )
  
  model_cv = xgb.cv(
    params = param0
    , nrounds = 1000
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
  
  model[[1]] = xgb.train(nrounds = iter, 
                         params = param0, 
                         data = xgtrain, 
                         watchlist = watchlist, 
                         print.every.n = 100,
                         nthread = 6,
                         verbose = 1)
  
  predicted[[1]] <- predict(model[[1]], xgval)
  cat(auc(actual, predicted[[1]]),"\n")
  spred <- spred + predicted[[1]]
}  
spred <- spred/10
cat(k,auc(actual, spred),"<====== \n")

df.predicted <- data.frame(actual,predicted)
colnames(df.predicted) <- c("target","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10")

pt <- rowMeans(df.predicted)

df.predicted <- data.frame(df.predicted,pt)

cat(k,auc(actual, df.predicted$pt),"<====== \n")

xgsub <- xgb.DMatrix(data.matrix(test.set[,fields.00]))

predicted.s <- list()

for(i in 1:2){
  predicted.s[[i]] <- predict(model[[i]], xgsub)
}

df.predicted.s <- data.frame(predicted.s)
#colnames(df.predicted.s) <- c("p1","p2","p3","p4","p5","p6","p7","p8","p9","p10")
pt <- rowMeans(df.predicted.s)
#pt1 <- rowMedians(as.matrix((df.predicted.s)))
#df.predicted.s <- data.frame(df.predicted.s,pt,pt1)
#write.csv(df.predicted.s,"dfpredicted.csv",row.names = F)

submission <- read.table("test/sample_submission.csv", header=TRUE, sep=',')

#px <- ifelse(df.predicted.s$pt > 0.6,df.predicted.s$pt,ifelse(df.predicted.s$pt > 0.5,df.predicted.s$pt-0.2,df.predicted.s$pt ))

submission$TARGET <- df.predicted.s$pt
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

