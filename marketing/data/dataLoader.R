?marketing
loadData = function(){
  df = marketing
  
  #===SPACE CONTINUOUS VALUES PROPORTIONALLY====
  #income
  #incomes = c(0,10,15,20,25,30,40,50,75,75)
  incomes = 1:9
  #IncomeProp=sapply(df$Income,function (x) mean(c(incomes[x],incomes[x+1])))
  IncomeProp=sapply(df$Income,function (x) incomes[x])
  
  #age
  ages = c(15.5,21,29.5,39.5,49.5,59.5,70)
  AgeProp=sapply(df$Age,function (x) ages[x])
  
  #years
  years = c(.5,2,5,8.5,10)
  LivedProp=sapply(df$Lived,function (x) years[x])
  
  #===SPLIT INTO CATEGORICAL AND CONTINUOUS
  categorical = data.frame(factor(df$Occupation),factor(df$Sex),factor(df$Marital),factor(df$Dual_Income),factor(df$Status),factor(df$Home_Type),factor(df$Ethnic),factor(df$Language))
  continuous  = data.frame(IncomeProp,AgeProp,LivedProp,Edu,Household,Householdu18)
  
  #===IMPUTE DATA===
  miceCat  = mice(categorical,m=1,maxit=5,meth='polyreg',seed=300)
  miceCont = mice(continuous,m=1,maxit=5,meth='pmm',seed=300)
  
  catImpute = complete(miceCat,1)
  contImpute = complete(miceCont,1)
  imputed = cbind(catImpute,contImpute)
  colnames(imputed) = c("Occupation","Sex","Marital","Dual_Income","Status","Home_Type","Ethnic","Language","Income","Age","Lived","Edu","Household","Householdu18")
  attach(imputed)
  
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
  OccupationCat = expandCat(Occupation,c("Managerial","Sales","Laborer","Cleric","Homemaker","Student","Military","Retired"))
  MaritalCat = expandCat(Marital,c("Married","Co-habitate","Divorced","Widowed"))
  #EduCat = expandCat(Edu,c("NoHigh","SomeHigh","GradHigh","SomeColl","GradColl"))
  DualCat = expandCat(Dual_Income,c("Not Married,Dual","DualIncome"))
  StatusCat = expandCat(Status,c("Own","Rent"))
  Home_TypeCat=expandCat(Home_Type,c("House","Condo","Apartment","MobileHome"))
  EthnicCat=expandCat(Ethnic,c("American Indian","Asian","Black","EastIndian","Hispanic","Pacific Islander","White"))
  LanguageCat=expandCat(Language,c("English","Spanish"))
  
  #===REATTACH DATA AND RETURN===
  trimDF = data.frame(Income,Sex,MaritalCat,Age,Edu,OccupationCat,Lived,DualCat,Household,Householdu18,StatusCat,Home_TypeCat,EthnicCat,LanguageCat)
  return(trimDF)
}
catMarketing = loadData()
cache('catMarketing')
rm(loadData)
