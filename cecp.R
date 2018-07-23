# Function to compute Complexity-Entropy Causality Plane (CECP)
 # Input(2 arguments. Null arguments are not vaild)
  # x = Time series (type = numeric vector)
  # dim =  Embedding dimension (type = numeric)
 # Output is a list. List contains ordinal pattern distribution, 
 # normalized permutation entropy, and complexity

cecp<-function(x,dim){
  
op<-ordinal_pattern(x,dim)
npe<-permu_entropy(op)
comp_JS<-complexity(op)

result_list<-list(op=op,npe=npe,complexity=comp_JS)
return(result_list)
}