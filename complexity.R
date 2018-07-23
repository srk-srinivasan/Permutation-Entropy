# Function to compute complexity of  a given time series
  # Input (1 argument. Null argument not valid)
   # op = Ordinal pattern computed using the function ordinal_pattern 
   # op (type=numeric vector)
  # Output is complexity (type=numeric)

complexity<-function(op){

# Comp_JS = Q_o * JSdivergence * pe  
# Normalizing constant Q_o
# JSdivergence = Jensen-Shannon divergence
# pe = permutation entopry  

pe<-permu_entropy(op)    

constant1<- (0.5+((1 - 0.5)/length(op)))*log(0.5+((1 - 0.5)/length(op)))
constant2<-((1 - 0.5)/length(op))*log((1 - 0.5)/length(op))*(length(op) - 1) 
constant3<- 0.5*log(length(op))
Q_o<- -1/(constant1+constant2+constant3)

temp_op_prob<-op/sum(op)
temp_op_prob2<-(0.5*temp_op_prob)+(0.5*(1/length(op)))
JSdivergence<-(entropy::entropy(temp_op_prob2) - 0.5 * entropy::entropy(temp_op_prob) - 0.5 * log(length(op)))

Comp_JS = Q_o * JSdivergence * pe
return(Comp_JS)

}
