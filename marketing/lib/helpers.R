missingPlot = function(df){
  ggr_plot <- aggr(df, col=c('navyblue','red'), numbers=TRUE,
                   sortVars=TRUE, labels=names(df), cex.axis=.7,
                   gap=3, ylab=c("Histogram of missing data","Pattern"))
}