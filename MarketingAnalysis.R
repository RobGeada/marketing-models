library("ElemStatLearn")
library("rjags")
?marketing
str(marketing)
head(marketing)

#===FACTOR OUT CATEGORICALS INTO DIFFERENT FIELDS===
trimDF = marketing[complete.cases(marketing),]
attach(trimDF)

curriedColumn = function(column){
  return(function(i) as.integer(column==i))
} 

expandCat = function(column,names){
  values = 1:length(names)
  colSelector = curriedColumn(column)
  df = data.frame(sapply(values,colSelector))
  colnames(df)=names
  return(df)
}

train=sample(dim(trimDF)[1],5000)
OccupationCat = expandCat(Occupation,c("Managerial","Sales","Laborer","Cleric","Homemaker","Student","Military","Retired"))
MaritalCat = expandCat(Marital,c("Married","Co-habitate","Divorced","Widowed"))
EduCat = expandCat(Edu,c("NoHigh","SomeHigh","GradHigh","SomeColl","GradColl"))
DualCat = expandCat(Dual_Income,c("Not Married,Dual","DualIncome"))
StatusCat = expandCat(Status,c("Own","Rent"))
Home_TypeCat=expandCat(Home_Type,c("House","Condo","Apartment","MobileHome"))
EthnicCat=expandCat(Ethnic,c("American Indian","Asian","Black","EastIndian","Hispanic","Pacific Islander","White"))
LanguageCat=expandCat(Language,c("English","Spanish"))

#===SPACE INCOME PROPORTIONALLY===
incomes = c(5,12.5,17.5,22.5,27.5,35,45,62.5,100)
IncomeProp=sapply(Income,function (x) incomes[x])

#===SPACE AGE PROPORTIONALLY===
ages = c(15.5,21,29.5,39.5,49.5,59.5,70)
AgeProp=sapply(Age,function (x) ages[x])

#===SPACE YEARS PROPORTIONALLY===
years = c(.5,2,5,8.5,12)
LivedProp=sapply(Lived,function (x) years[x])

#===REATTACH DATA===
trimDF = data.frame(IncomeProp,Sex,MaritalCat,AgeProp,EduCat,OccupationCat,LivedProp,DualCat,Household,Householdu18,StatusCat,Home_TypeCat,EthnicCat,LanguageCat)
trimDF

#===SEPERATE INTO TRAIN AND TEST SETS
trainDF = trimDF[train,]
testDF  = trimDF[-train,]

row.names(trainDF) = NULL

lm.fit=lm(IncomeProp~.,data=trainDF)

summary(lm.fit)
y=predict(lm.fit,newdata=testDF)
mean(abs(testDF$Income-y))
sd(testDF$Income-y)
hist(testDF$Income-y)
summary(lm.fit)

#===BAYESIAN LINEAR MODEL===

data = list(y = IncomeProp, X = trainDF[,-1],n = nrow(trainDF),p = ncol(trainDF[,-1]))

modelString = 
  "model {
    for(i in 1:n) {
      y[i] ~ dnorm(mu[i], tau)
      mu[i] <- beta0 + inprod(beta[], X[i,])
    }
    tau ~ dgamma(0.01,0.01)
    beta0 ~ dnorm(0,1.0E-12)
    taub ~ dgamma(0.01,0.01)
    for (j in 1:p) {
      ind[j]~dbern(0.2)
      betaT[j]~dnorm(0,taub)
      beta[j]<-ind[j]*betaT[j] 
    }
  }"

lm.info = lm(IncomeProp ~ .,data=trainDF)  
beta.init = lm.info$coefficients[-1] #Use initial values predicted from frequentist model.
tau.init = length(IncomeProp)/sum(lm.info$residuals^2)
initsList = list(beta0 = 0, betaT = beta.init, tau = tau.init,ind = rep(0,39),taub=1)

model=jags.model(textConnection(modelString), data=data, inits=initsList)
update(model,n.iter=1000)
output=coda.samples(model=model,variable.names=c("beta0", "beta", "tau","ind"),n.iter=10000, thin=1)

plot(output)
autocorr.plot(output)
summary(output)
crosscorr(output)
effectiveSize(output)
crosscorr.plot(output)
pairs(as.matrix(output),pch=".")

colnames(trainDF)[11]

