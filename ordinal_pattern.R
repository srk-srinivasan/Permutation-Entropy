# Function to compute the ordinal patterns for a given time series.
 # Input (2 arguments. Null arguments are not vaild)
   # x = Given time series (type=numeric vector)
   # dim = Embedding dimension (type=numeric)
   # Commonly used value of dim ranges from 3 to 7 
 # Output is a numeric vector of size=(dim)!

ordinal_pattern<-function(x,dim){ 

# Generate ordinal numbers to assign. For example if dim =3, then 
# ordinal number=0,1,2  
ordinal_numbers<-seq(0,(dim-1),by=1)

# Compute all possible permutations of the ordinal numbers. 
# Maximum size of possible_pattern=dim!
possible_pattern<-(combinat::permn(ordinal_numbers))

# Initialize result. Result is the output. 
result<-0
result[1:length(possible_pattern)]<-0

# Loop for computation of ordinal pattern
for(i in 1:(length(x)-(dim-1))){
    temp<-x[i:(i+(dim-1))]
    tempseq<-seq(0,dim-1,by=1)
    tempdata<-data.frame(temp,tempseq)
    tempdata<-tempdata[order(temp),]
  
    for(j in 1: length(possible_pattern)){
        if (all(possible_pattern[[j]]==tempdata$tempseq)){
            result[j]<-result[j]+1
            }
    
       }
  
    }

return(result)

}



