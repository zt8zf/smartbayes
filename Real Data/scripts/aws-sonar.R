# November 30, 2025
# Code to run the sonar experiment on AWS
################################################################################
# Load libraries
library(KernSmooth)
library(locfit)
library(glmnet)
library(doParallel)
library(parallel)
library(MASS)
library(pracma)
library(naivebayes)
library(readr)
library(pROC)
library(mgcv)
library(dplyr)
library(clustermq)

################################################################################

# Set clustermq to use multicore scheduler
options(clustermq.scheduler = "multicore")

# Print some introductory statements
cat("File lr/nb/sb comparison: sonar data\n")
cat("Using spline density ratio estimate\n")

# Load and preprocess data
# Users should change this to point to wherever the data are stored.
sonar <- read_csv(here("Real Data/data/sonar.all-data"))

# Create binary variable
yval <- ifelse(sonar$X61 == "M", 1, 0)

# Drop character column
sonar$X61 <- NULL

# Create a variable for all the sonar data
all.funcs <- as.matrix(na.omit(sonar))

# Compute the number of rows and number of columns in the dataset
num.rows <- nrow(all.funcs)
d <- ncol(all.funcs)

# Recursive sample function ensuring at least one of each class
get.samp.recur <- function(num.rows, m, yval){
  samp1 <- sample.int(num.rows, m, replace = FALSE)
  num.y <- n_distinct(yval[samp1])
  min.class <- min(table(yval[samp1]))
  if(num.y < 2 | min.class < 2){
    samp1 <- get.samp.recur(num.rows, m, yval)
  }
  return(samp1)
}

# Function to compute the dim or length of a class
dimorlen <- function(x){
  if (is.null(nrow(x))) length(x) else nrow(x)
}

# Spline density ratio function
dr1 <- function(x0, x1, y, xtest) {
  d <- ncol(x0)
  tmp <- matrix(0, nrow(x0)+nrow(x1), d)
  tmptest <- matrix(0, nrow(xtest), d)
  
  for (j in 1:d) {
    z <- c(x0[,j], x1[,j])
    tmp1 <- tryCatch(gam(y ~ s(z), family = "binomial"),
                     error = function(e){
                       total.val <- length(unique(z))
                       tryCatch(gam(y ~ s(z, k = total.val), family = "binomial"),
                                error = function(e) NULL)
                     })
    if (!is.null(tmp1)){
      tmp[, j] <- predict(tmp1, newdata = data.frame(z = z))
      tmptest[, j] <- predict(tmp1, newdata = data.frame(z = xtest[, j]))
    }
  }
  list(drtrain = tmp - log(dimorlen(x0)/dimorlen(x1)),
       drtest = tmptest - log(dimorlen(x0)/dimorlen(x1)))
}

# Define training sizes for the experiment
m.seq <- seq(80, 206, by = 4)

# Number of iterations
num.i <- 200

# This is the main function to run the experiment
run_sonar_function <- function(rep_idx){
  log_file <- paste0("worker_log_", Sys.getpid(), ".txt")
  set.seed(rep_idx + 1000L)
  
  # Define a matrix to store the results from each iteration
  imat <- matrix(NA, nrow = length(m.seq), ncol = 3)
  
  for (mm in seq_along(m.seq)){
    m <- m.seq[mm]
    
    # Get training and testing sample
    train.samp <- get.samp.recur(num.rows, m, yval)
    test.samp <- setdiff(1:num.rows, train.samp)
    
    # Get x and y training data
    xtrain <- as.matrix(ion[train.samp,])
    ytrain <- yval[train.samp]
    
    # Get testing data
    xtest <- as.matrix(ion[test.samp,])
    ytest <- yval[test.samp]
    
    # GLM
    glm1 <- glm(ytrain ~ xtrain, family = "binomial")
    pred.glm <- tryCatch({
      predict(glm1, newdata = data.frame(xtrain = xtest), type = "response")
    }, error = function(e) NA)
    mis.glm <- mean(ytest != (pred.glm > 0.5), na.rm = TRUE)
    
    # Naive Bayes
    nb <- naive_bayes(xtrain, as.factor(ytrain), usekernel = TRUE)
    pred.nb <- predict(nb, xtest, type = "class")
    mis.nb <- mean(ytest != pred.nb)
    
    # Smart Bayes
    x0 <- xtrain[ytrain == 0,]
    x1 <- xtrain[ytrain == 1,]
    
    # Define training smart bayes y-values
    ytrain_sb <- c(rep(0, nrow(x0)), rep(1, nrow(x1)))
    
    # Compute the density ratios
    sb.data <- dr1(x0, x1, ytrain_sb, xtest)
    ratios.train <- sb.data$drtrain
    ratios.test <- sb.data$drtest
    
    # Build the smart bayes model
    sb.full <- glm(ytrain_sb ~ ratios.train, family = "binomial")
    d1.pred <- predict(sb.full, newdata = data.frame(ratios.train = ratios.test), type = "response")
    mis.d1 <- mean(ytest != (d1.pred > 0.5))
    
    imat[mm, ] <- c(mis.glm, mis.nb, mis.d1)
    cat(paste0("Finished m = ", m, "\n"), file = log_file, append = TRUE)
  }
  
  data.frame(
    rep = rep_idx,
    m = m.seq,
    glm_error = imat[, 1],
    naive_bayes_error = imat[, 2],
    sb_error = imat[, 3]
  )
}

# Parallel execution
result_list <- Q(run_sonar_function,
                 rep_idx = seq_len(num.i),
                 n_jobs = parallel::detectCores())

result_df <- bind_rows(result_list)

# Summary
summary_df <- result_df %>%
  group_by(m) %>%
  summarize(
    glm_mean = mean(glm_error, na.rm = TRUE),
    glm_sd   = sd(glm_error, na.rm = TRUE),
    nb_mean  = mean(naive_bayes_error, na.rm = TRUE),
    nb_sd    = sd(naive_bayes_error, na.rm = TRUE),
    sb_mean = mean(sb_error, na.rm = TRUE),
    sb_sd   = sd(sb_error, na.rm = TRUE),
    .groups = "drop"
  )

# Save results
write.csv(summary_df, paste0("../output_data/aws-sonar-summary-", num.i, ".csv"), row.names = FALSE)
saveRDS(summary_df, paste0("../output_data/aws-sonar-summary-", num.i, ".RDS"))
