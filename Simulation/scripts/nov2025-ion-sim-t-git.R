################################################################################
# November 30, 2025
# Code for the multivariate t data simulation using the ionosphere parameters
################################################################################
# Load libraries
library(dplyr)
library(naivebayes)
library(mgcv)
library(readr)
library(readxl)
library(dplyr)
library(naivebayes)
library(mgcv)
library(MultiRNG)

################################################################################

# Read in data
ion <- read_csv(here("Real Data/data/ionosphere.data"))

# Discard other columns
ion$X1 <- NULL
ion$X2 <- NULL
colnames(ion)

# Make y value
yval <- ifelse(ion$X35=="b",0,1)

# Discard extra variable
ion$X35 <- NULL

################################################################################
# Get sample means and covariances from ionosphere dataset
ion0 <- ion[which(yval == 0),]
ion1 <- ion[which(yval == 1),]

ion.mu0 <- apply(ion0, 2, mean)
ion.mu1 <- apply(ion1, 2, mean)

ion.cov <- cov(ion)
ion.cov0 <- cov(ion0)
ion.cov1 <- cov(ion1)
################################################################################
# Add functions we need
# Dim or len function
dimorlen <- function(x){
  if (is.null(nrow(x))==T){
    out <- length(x)
  } else {
    out <- nrow(x)
  }
  return(out)
}

# This dr1 function works when d > 1
dr1 <- function(x1,x2,y,xtest) {
  d=ncol(x1)
  tmp=matrix(NA,nrow(x1)+nrow(x2),d)
  tmptest=matrix(NA,nrow(xtest),d)
  for (i in 1:d) {
    z=c(x1[,i],x2[,i])
    tmp1=gam(y~s(z), family="binomial")
    tmp[,i]=predict(tmp1,newdata=data.frame(z=z))
    tmptest[,i]=predict(tmp1,newdata=data.frame(z=xtest[,i]))
  }
  list(drtrain=tmp-log(dimorlen(x1)/dimorlen(x2)),drtest=tmptest-log(dimorlen(x1)/dimorlen(x2)))
}

################################################################################

# Define number of methods in the simulation
num.methods <- 3

# Number of repetitions at a given sample size
num.i <- 100
imat <- matrix(NA, nrow = num.i, ncol = num.methods)

# Number of different sample sizes
qout <- matrix(NA, nrow = 30, ncol = num.methods)

# Define number of covariates
num.p <- length(ion.mu0)

# Predefine vector of training sizes
qvec <- 120 + 30*(1 : nrow(qout)-1)

################################################################################
# Degrees of freedom to simulate
t.df.vec <- c(5, 10, 30)

# Loop over the different degrees of freedom
for (t.df in t.df.vec) {
  
  # Loop over the qvec
  for (q in 1:nrow(qout)) {
    
    # Define training size
    m <- qvec[q]
    
    # Set seed
    set.seed(m)
    
    # Print update
    cat("t.df:", t.df, "m:", m, "\n")
    
    for (i in 1:num.i) {
      
      # Generate class 0 data
      class0 <-
        mvtnorm::rmvt(n = m,
                      df = t.df,
                      delta = ion.mu0,
                      sigma = ion.cov0)
      
      # Generate class 1 data
      class1 <-
        mvtnorm::rmvt(n = m,
                      df = t.df,
                      delta = ion.mu1,
                      sigma = ion.cov1)
      
      
      # Assign class names
      colnames(class1) <- colnames(class0) <- paste0("X", 1:num.p)
      
      # Make all functions
      all.funcs <- rbind(class0, class1)
      num.rows <- nrow(all.funcs)
      
      # Make y val
      yval <- c(rep(0, m), rep(1, m))
      
      # Divide into train and test
      train.samp <- sample(x = 1:num.rows,
                           size = m,
                           replace = F)
      test.samp <- setdiff(1:num.rows, train.samp)
      
      # Get x and y training data
      xtrain <- as.matrix(all.funcs[train.samp, ])
      ytrain <- yval[train.samp]
      
      
      # Get testing data
      xtest <- as.matrix(all.funcs[test.samp, ])
      ytest <- yval[test.samp]
      
      # Train glm
      t1 <- xtrain
      glm1 <- glm(ytrain ~ t1, family = "binomial")
      
      # Test glm
      t1 <- xtest
      pred.glm <-
        predict(object = glm1,
                newdata = data.frame(t1 = t1),
                type = "response")
      mis.glm <- sum(ytest != (pred.glm > 0.5))
      
      ############################################################################
      
      # Train NB
      nb <- naive_bayes(xtrain, as.factor(ytrain), usekernel = T)
      
      # Test NB
      pred.nb2 <- predict(nb, xtest, type = "class")
      mis.nb <- sum(ytest != pred.nb2)
      
      ############################################################################################
      # Get sb data
      x0 <- xtrain[ytrain == 0,]
      x1 <- xtrain[ytrain == 1,]
      
      num1 <- sum(ytrain)
      num0 <- length(ytrain) - num1
      ytrain <- c(rep(0, num0), rep(1, num1))
      
      sb.data <- dr1(x0, x1, ytrain, xtest)
      
      if (is.null(sb.data)) {
        next
      }
      
      ratios.train <- sb.data$drtrain
      ratios.test <- sb.data$drtest
      
      # Train model
      d1 <- ratios.train
      sb.full <- glm(ytrain ~ d1, family = "binomial")
      
      # Test model
      d1 <- ratios.test
      d1.pred <-
        predict(sb.full, newdata = data.frame(d1 = d1), type = "response")
      mis.d1 <- sum(ytest != 1 * (d1.pred > 0.5))
      
      ############################################################################
      
      # Record results
      imat[i, 1] <- mis.glm / length(ytest)
      imat[i, 2] <- mis.nb / length(ytest)
      imat[i, 3] <- mis.d1 / length(ytest)
    }
    
    qout[q, ] <- apply(imat, 2, mean)
    cat("t.df:", t.df, "q:", q, qout[q, ], "\n")
    
  }
  
  saveRDS(qout, file = paste0("p32-t-ionosphere-df", t.df, "-m990.RDS"))
  
}