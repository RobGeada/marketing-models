#===PROJECT SETUP===
setwd("~/Documents/GradSchool/FirstYear/StatsForBigData/Project/marketing")
load.project()

#===SEPERATE INTO TRAIN AND TEST SETS
trainDF = trimDF[train,]
testDF  = trimDF[-train,]

#==LINEAR PREDICTION===
lmfit = lm(IncomeProp~.,data=trainDF)
par(mfrow=c(2,2))
#plot(lmfit)
predictions=predict(lmfit,newdata=testDF)
errors[i]=mean(abs((predictions)-testDF$IncomeProp)/testDF$IncomeProp)
mean(errors)


#===PCR PREDICTIONS
library(pls)
pcrFit = pcr(IncomeProp~.,data=trainDF,scale=TRUE,validation="CV")
summary(pcrFit)
plot(pcrFit)
predictions=predict(pcrFit,newdata=testDF,ncomp=25)
mean(abs(predictions-testDF$IncomeProp)/testDF$IncomeProp)

write.csv(trimDF,"marketingTrim.csv")
#trimDF

