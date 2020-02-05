rm(list = ls())
data=read.csv("./Data and Description-1575633872518/Train.csv")
head(data)
names(data)
sum(is.na(data[names(data)]))
summary(data)
library (DMwR)
#proportion of columns having NAs in each row
manyNAs(data,0.2)
manyNAs(data,0.1)
dim(data)
#checking for duplicated data
data=data[!duplicated(data),]
dim(data)
#checking for near zero variance columns
library(caret)
zerovar_col=nearZeroVar(data,names = TRUE)
zerovar_col
#checking the distribution of data
library(ggplot2)
par(mfrow=c(1,1))
boxplot(data$Attr1,
        main = "Box Plot: Attr1",
        ylab = "Attr1")
#plotting correlation matrix
library(corrplot)
corMat <- cor(data[,sapply(data, is.numeric)])
corrplot::corrplot(corMat)
corrplot::corrplot(corMat,tl.cex = 0.7)
corrplot::corrplot(corMat, tl.cex = 0.7, method = "number")
#checking the imbalance in data
barplot(table(data$target))
data2<-data
#impute nas 
#data2<-centralImputation(data2)
sum(is.na(data2))
boxplot(data2$Attr1)
class(boxplot(data2$Attr1))
names(boxplot(data2$Attr1))
#checking for length of outliers
Attr3_out<-which(data2$Attr3%in%boxplot(data$Attr3,plot = FALSE)$out)
length(Attr3_out)
#quantile clipping
data2$Attr3[(data2$Attr3 < quantile(data2$Attr3,0.01,na.rm = TRUE))]<- quantile(data2$Attr3,0.01,na.rm = TRUE)
data2$Attr2[(data2$Attr2 > quantile(data2$Attr2,0.99,na.rm = TRUE))]<- quantile(data2$Attr2,0.99,na.rm = TRUE)
names(data2)
data2=data2[,-65]
for(i in names(data2)){
  data2[i][(data2[i] < quantile(data2[i],0.01,na.rm = TRUE))]<- quantile(data2[i],0.01,na.rm = TRUE)
  data2[i][(data2[i] > quantile(data2[i],0.99,na.rm = TRUE))]<- quantile(data2[i],0.99,na.rm = TRUE)
  
}
range(data$Attr5,na.rm = TRUE)
range(data2$Attr5,na.rm = TRUE)
sum(is.na(data2))

# tuning and tewaking little more 
library(reshape2)
## REMOVING HIGHLY Correlated features


cor_mat <- cor(data2,use = "pairwise.complete.obs")
sum(is.na(cor_mat))
cor_mat[lower.tri(cor_mat)] <- NA
cor_mat
sum(is.na(cor_mat))
cor_mat1 <- na.omit(melt(cor_mat))
cor_mat1
cor_mat1 <- cor_mat1[cor_mat1$Var1!=cor_mat1$Var2,]
cor_mat1
cor_mat1 <- cor_mat1[abs(cor_mat1$value)>0.7,]
cor_mat1
to_remove_features <- unique(as.character(cor_mat1$Var1))
to_remove_features

req_cols<-setdiff(names(data2),to_remove_features)

data2<-data2[,req_cols]
names(data2)
sum(is.na(data2))
data2[req_cols] <- apply(data2[req_cols], 2, function(x){x[is.na(x)]=mean(x,na.rm = T)})
sum(is.na(data2))
length(req_cols)
#standardising data using Zscore 
library(vegan)
data_std2<-decostand(data2,"standardize")
tar<-data[,65]
names(tar)
data_mod2<-cbind(data_std2,tar)
names(data_mod2)
library(caret)
set.seed(124)
train_rows<-createDataPartition(data_mod2$tar,p=0.8,list = F)
train1_data<-data_mod2[train_rows,]
val1_data<-data_mod2[-train_rows,]
dim(train1_data)
dim(val1_data)
table(train1_data$tar)/nrow(train1_data)
table(val1_data$tar)/nrow(val1_data)
#building vanilla model
log_reg2<-glm(tar~.,data = train1_data,family = "binomial")
summary(log_reg2)
prob_train<-predict(log_reg2,data=train1_data,type = "response")
#prob_val<-predict(log_reg1,data=val1_data,type = "response")
names(train1_data)
names(log_reg2)

install.packages("ggplot")
library(ROCR)
pred<-prediction(prob_train,train1_data$tar)
pred
names(pred)
perf<-performance(pred,measure = "tpr",x.measure = "fpr")
plot(perf,col=rainbow(1),colorise=T,print.cutoffs.at=seq(0,1,0.001))
perf_auc<-performance(pred,measure = "auc")
perf_auc

prob_val<-predict(log_reg2,val1_data,type = "response")
nrow(val1_data)
length(prob_val)
prob_class_val<-ifelse(prob_val>0.3,"yes","no")
table(prob_class_val)/nrow(val1_data)

prob_class_val<-as.factor(prob_class_val)
conf_matrix <- table(val1_data$target, prob_class_val)
print(conf_matrix)
str(prob_class_val)
