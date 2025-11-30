# Liver dataset experiment
################################################################################
# Load libraries
library(KernSmooth)
library(locfit)
library(glmnet)
library(R.utils)
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
# Set clustermq to multicore
options(clustermq.scheduler = "multicore")

cat("File lr/nb/sb comparison: liver data\n")
cat("Using spline density ratio estimate\n")

################################################################################
# Load and preprocess data
# Users should change this to point to wherever the data are stored.
liver <- read_csv(here("Real Data/data/Indian Liver Patient Dataset (ILPD).csv"))
liv <- data.frame(liver)

# Outcome variable
yval <- liv$X11 - 1

# Drop non-predictor columns
all.funcs <- na.omit(liv)
all.funcs$X2 <- NULL
all.funcs$X11 <- NULL

# Get dimensions of dataset
d <- ncol(all.funcs)
num.rows <- nrow(all.funcs)

################################################################################
# Function definitions
dimorlen <- function(x) {
  if (is.null(nrow(x))) length(x) else nrow(x)
}

# Spline density ratio function
dr1 <- function(x0, x1, y, xtest) {
  d <- ncol(x0)
  tmp <- matrix(NA, nrow(x0) + nrow(x1), d)
  tmptest <- matrix(NA, nrow(xtest), d)
  
  for (j in 1:d) {
    z <- c(x0[, j], x1[, j])
    tmp1 <- gam(y ~ s(z), family = "binomial")
    
    tmp[, j] <- predict(tmp1, newdata = data.frame(z = z))
    tmptest[, j] <- predict(tmp1, newdata = data.frame(z = xtest[, j]))
  }
  
  list(
    drtrain = tmp - log(dimorlen(x0)/dimorlen(x1)),
    drtest  = tmptest - log(dimorlen(x0)/dimorlen(x1))
  )
}

################################################################################
# Running the experiment

# Define training sizes
m.seq <- seq(50, 582, by = 8)

# Define number of repetitions at each training size
num.i <- 200

# This is the main liver function
run_liver_function <- function(rep_idx) {
  set.seed(rep_idx + 1000L)
  imat <- matrix(NA, nrow = length(m.seq), ncol = 3)
  
  for (mm in seq_along(m.seq)) {
    m <- m.seq[mm]
    
    # Training and test samples
    train.samp <- sample(x = 1:num.rows, size = m, replace = FALSE)
    test.samp  <- setdiff(1:num.rows, train.samp)
    
    xtrain <- as.matrix(all.funcs[train.samp, ])
    ytrain <- yval[train.samp]
    xtest  <- as.matrix(all.funcs[test.samp, ])
    ytest  <- yval[test.samp]
    
    # GLM
    glm1 <- glm(ytrain ~ xtrain, family = "binomial")
    pred.glm <- predict(glm1, newdata = data.frame(xtrain = xtest), type = "response")
    mis.glm <- mean(ytest != (pred.glm > 0.5))
    
    # Naive Bayes
    nb <- naive_bayes(xtrain, as.factor(ytrain), usekernel = TRUE)
    pred.nb <- predict(nb, xtest, type = "class")
    mis.nb <- mean(ytest != pred.nb)
    
    ################################################################
    # Smart Bayes (SB)
    x0 <- xtrain[ytrain == 0, ]
    x1 <- xtrain[ytrain == 1, ]
    ytrain_sb <- c(rep(0, nrow(x0)), rep(1, nrow(x1)))
    
    # Compute the density ratios
    sb.data <- dr1(x0, x1, ytrain_sb, xtest)
    ratios.train <- sb.data$drtrain
    ratios.test <- sb.data$drtest
    
    # Build the smart bayes model
    sb.full <- glm(ytrain_sb ~ ratios.train, family = "binomial")
    pred.sb <- predict(sb.full, newdata = data.frame(ratios.train = ratios.test), type = "response")
    mis.sb <- mean(ytest != (pred.sb > 0.5))
    
    # Record results
    imat[mm, ] <- c(mis.glm, mis.nb, mis.sb)
  }
  
  data.frame(
    rep = rep_idx,
    m = m.seq,
    glm_error = imat[, 1],
    naive_bayes_error = imat[, 2],
    sb_error = imat[, 3]
  )
}

################################################################################
# Post-processing

result_list <- Q(run_liver_function,
                 rep_idx = seq_len(num.i),
                 n_jobs = parallel::detectCores())

result_df <- bind_rows(result_list)

summary_df <- result_df %>%
  group_by(m) %>%
  summarize(
    glm_mean = mean(glm_error, na.rm = TRUE),
    glm_sd   = sd(glm_error, na.rm = TRUE),
    nb_mean  = mean(naive_bayes_error, na.rm = TRUE),
    nb_sd    = sd(naive_bayes_error, na.rm = TRUE),
    sb_mean  = mean(sb_error, na.rm = TRUE),
    sb_sd    = sd(sb_error, na.rm = TRUE),
    .groups = "drop"
  )

# ---- Save ----
write.csv(summary_df,
          paste0("../output_data/aws-liver-summary-", num.i, ".csv"),
          row.names = FALSE)
saveRDS(summary_df,
        paste0("../output_data/aws-liver-summary-", num.i, ".RDS"))
