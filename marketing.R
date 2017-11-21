library("ElemStatLearn")
library(leaps)
?marketing

#===REMOVE NA FIELDS===
df = marketing

#===MEAN IMPUTATION==+
nummode = function(v) {
  uniques = unique(v)
  uniques[which.max(tabulate(match(v, uniqv)))]
}

for (name in colnames(df)){
  df[,name][is.na(df[,name])]=nummode(df[,name])
}
?mode
attach(df)
df
#===DEFINE CATEGORICAL EXPANSION FUNCTIONS===
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

#===FACTOR OUT CATEGORICALS INTO DIFFERENT FIELDS===
train=sample(dim(df)[1],7900)
OccupationCat = expandCat(Occupation,c("Managerial","Sales","Laborer","Cleric","Homemaker","Student","Military","Retired"))
MaritalCat = expandCat(Marital,c("Married","Co-habitate","Divorced","Widowed"))
EduCat = expandCat(Edu,c("NoHigh","SomeHigh","GradHigh","SomeColl","GradColl"))
DualCat = expandCat(Dual_Income,c("Not Married,Dual","DualIncome"))
StatusCat = expandCat(Status,c("Own","Rent"))
Home_TypeCat=expandCat(Home_Type,c("House","Condo","Apartment","MobileHome"))
EthnicCat=expandCat(Ethnic,c("American Indian","Asian","Black","EastIndian","Hispanic","Pacific Islander","White"))
LanguageCat=expandCat(Language,c("English","Spanish"))

#===SPACE INCOME PROPORTIONALLY===
incomes = c(5,12.5,17.5,22.5,27.5,35,45,62.5,75)
#incomes=c(1,2,3,4,5,6,7,8,9)
IncomeProp=sapply(Income,function (x) incomes[x])

#===SPACE AGE PROPORTIONALLY===
ages = c(15.5,21,29.5,39.5,49.5,59.5,70)
AgeProp=sapply(Age,function (x) ages[x])

#===SPACE YEARS PROPORTIONALLY===
years = c(.5,2,5,8.5,12)
LivedProp=sapply(Lived,function (x) years[x])

#===REATTACH DATA===
trimDF = data.frame(IncomeProp,Sex,MaritalCat,AgeProp,EduCat,OccupationCat,LivedProp,DualCat,Household,Householdu18,StatusCat,Home_TypeCat,EthnicCat,LanguageCat)
dim(trimDF)

#===SEPERATE INTO TRAIN AND TEST SETS
trainDF = trimDF[train,]
testDF  = trimDF[-train,]

#==LINEAR PREDICTION===
lmfit = lm(log(IncomeProp)~.,data=trainDF)
par(mfrow=c(2,2))
plot(lmfit)
predictions=predict(lmfit,newdata=testDF)
mean(abs(exp(predictions)-testDF$IncomeProp))

#===PCR PREDICTIONS
library(pls)
pcrFit = pcr(IncomeProp~.,data=trainDF,scale=TRUE,validation="CV")
summary(pcrFit)
plot(pcrFit)
predictions=predict(pcrFit,newdata=testDF,ncomp=25)
mean(abs(predictions-testDF$IncomeProp))

write.csv(trimDF,"marketingTrim.csv")
trimDF
