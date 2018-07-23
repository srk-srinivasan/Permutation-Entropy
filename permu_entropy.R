# Function to compute permutation entropy of  a given time series
 # Input (1 argument, Null argument not valid)
   # op = Ordinal pattern computed using the function ordinal_pattern 
   # op (type=numeric vector)
 # Output is normalized permutation entropy (type=numeric)

permu_entropy<-function(op){
# Compute maximum entropy. maximum entropy = log(dim!)
# or maximum entropy = log(length(ordinal_pattern))
entropy_max<-log(length(op))

# Normalized permutation entropy
npe<-entropy::entropy(op)/entropy_max
return(npe)

}