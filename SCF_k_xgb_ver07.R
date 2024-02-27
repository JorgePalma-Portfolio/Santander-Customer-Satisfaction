library(Metrics)
library(xgboost)
library(caret)
library(cvTools)
library(digest)

group_into_buckets <- function(var,p){
  cut(var, 
      breaks= unique(quantile(var,probs=seq(0,1,by=p), na.rm=T)),
      include.lowest=T, ordered=T) 
}

psum <- function(...,na.rm=FALSE) { 
  rowSums(do.call(cbind,list(...)),na.rm=na.rm) }


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

all_data <- all_data[,apply(all_data[,-c(1:3)],2,sd) > 0]
all_data <- all_data[!duplicated(lapply(all_data, digest))]


all_data$numzero <- rowSums(all_data[,-c(1:3)] == 0)
#all_data$pvar38  <- as.numeric(group_into_buckets(all_data$var38,1/5))
#all_data$pvar15  <- as.numeric(group_into_buckets(all_data$var15,1/3))

rm(test,train)

all_data <- data.frame(all_data[with(all_data,order(ID)),])

all_data[all_data == -999999] <- 2

novariation.fields <- c("ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28", "ind_var27", "ind_var41", 
                        "ind_var46_0", "ind_var46", "num_var27_0", "num_var28_0", "num_var28", "num_var27", "num_var41", 
                        "num_var46_0", "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46", "imp_amort_var18_hace3",
                        "imp_amort_var34_hace3", "imp_reemb_var13_hace3", "imp_reemb_var33_hace3", "imp_trasp_var17_out_hace3", 
                        "imp_trasp_var33_out_hace3", "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3", "num_reemb_var33_hace3", 
                        "num_trasp_var17_out_hace3", "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3")

all_data <- all_data[,!(names(all_data) %in% novariation.fields)]

data_1 <- list()
data_1$custom0 <- log1p(all_data$var15)
data_1$custom1 <- log(all_data$var38)
all_data       <- data.frame(all_data,data_1) 

fields.00 <- -c(1:3)

train.set <- all_data[all_data$Train_Flag==1,] 
test.set  <- all_data[all_data$Train_Flag==0,]  



  train_70 <- train.set[,fields.00]
  train_30 <- train.set[,fields.00]
  y  <- train.set[,2]
  ny <- train.set[,2]
  
  xgtrain <- xgb.DMatrix(as.matrix(train_70), label = y)
  xgval <- xgb.DMatrix(as.matrix(train_30))
  watchlist <- list('train_70' = xgtrain)
  
  cctrl1 <- trainControl(method = "cv", number = 3,
                         classProbs = TRUE) 
  
  
  
  xgbTree_caret <- train(x = train_70,
                         y = y,
                         method = "xgbTree",
                         metric = "ROC", 
                         preProc = c("center", "scale"),
                         trControl = cctrl1 
                         )
  
  
