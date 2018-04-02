library(rJava)
library(iplots)
library(partitions)
library(Deducer)

KeRF_uf<-function(x,k)
{
  d=length(x)
  S=0
  if (k==1&d>k)
    temp=matrix(c(1,rep(0,d-k)),d,1)
  else
    temp=restrictedparts(k,d)
  p=NULL
  for(i in 1:dim(temp)[2])
    p=rbind(p,perm(temp[,i]))
  
  for(i in 1:dim(p)[1])
  {
    multiplier=factorial(k)/cumprod(factorial(p[i,]))[d]*(1/d)^k
    product=rep(NA,d)
    for (m in 1:d)
    {
      sum=0;
      for(j in 1:p[i,m]-1)
        if(j>-1)
          sum=sum+(-log(x[m]))^j/factorial(j)
      product[m]=1-x[m]*sum  
    }
    S=S+multiplier*cumprod(product)[d]
  }
  return(S)
}

k=2
N=100
x=seq(0,1,length.out = N)
y=seq(0,1,length.out = N)
Z=matrix(NA,N,N)
for (i in 1:N)
  for (j in 1:N)
  {
    cat("compute the value at location:",i/N,j/N,"\n")
    distance=abs(c(x[i]-0.5,y[j]-0.5))
    Z[i,j]=KeRF_uf(distance,k)
  }


library(RColorBrewer)
library(lattice)
grid <- expand.grid(X=x, Y=y)
dim(Z) <- c(N*N,1)
grid$Z <- Z

levelplot(Z ~ X*Y, grid, at=seq(0,1,length.out = 1000),xlab = NULL,ylab = NULL,
          col.regions=function(n)colorRampPalette(brewer.pal(9,"Blues"))(n))

source("~/Dropbox/utils.R")
par(mar = c(2,2,1,1))
plot(c(0, 1), c(0,1), type = "n", xlab = "", ylab = "")
abline(v=.4)
abline(v=.6)
abline(h=.4)
abline(h=.6)
rect(.4,.4,.6,.6, col="blue")
utils$save_pdf("~/papers/dl-stats/paper/fig/", "cylinder_kernel")

