#===PROJECT SETUP===
library(ProjectTemplate)
setwd("~/Documents/GradSchool/FirstYear/StatsForBigData/Project/marketing")
rm(list=ls())
load.project()

#===SEPERATE INTO TRAIN AND TEST SETS
train=sample(dim(catMarketing)[1],dim(catMarketing)[1]*.75)
trainDF = catMarketing[train,]
testDF  = catMarketing[-train,]

#==GRAB ACTUAL Y VALUES
par(mfrow=c(1,1))
y = log(testDF$Income)
hist(trainDF$Income)

#==LINEAR PREDICTION===
lmfit = lm(log(Income)~.,data=trainDF)
par(mfrow=c(2,2))
plot(lmfit)
summary(lmfit)
predictions=predict(lmfit,newdata=testDF)
errors=mean(abs((predictions-y)/y))
print(errors)

#==LINEAR PREDICTIONS DIAGNOSTICS
plotDF = data.frame(y,predictions)
ggplot(plotDF,aes(x=y,y=predictions)) +
  geom_point()+
  geom_segment(x=0,y=0,xend=10,yend=10)


#===PCR PREDICTIONS
library(pls)
pcrFit = pcr(log(Income)~.,data=trainDF,scale=TRUE,validation="CV")
summary(pcrFit)
plot(pcrFit)
predictions=predict(pcrFit,newdata=testDF,ncomp=22)
mean(abs(predictions-y)/y)

write.csv(catMarketing,"marketingTrim.csv")
#trimDF

