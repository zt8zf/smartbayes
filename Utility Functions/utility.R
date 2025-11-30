################################################################################
# November 30, 2025
# Utlity functions which are used in other scripts
################################################################################

# Recursive sampling function ensuring both classes are represented
get.samp.recur <- function(num.rows, m, yval) {
  samp1 <- sample.int(num.rows, m, replace = FALSE)
  if (m < 1000) {
    num.y <- n_distinct(yval[samp1])
    min.class <- min(table(yval[samp1]))
    if (num.y < 2 | min.class < 2) {
      samp1 <- get.samp.recur(num.rows, m, yval)
    }
  }
  samp1
}

# Function to compute dim or length
dimorlen <- function(x) {
  if (is.null(nrow(x))) length(x) else nrow(x)
}

# Spline density ratio function
dr1 <- function(x0, x1, y, xtest) {
  d <- ncol(x0)
  tmp <- matrix(0, nrow(x0) + nrow(x1), d)
  tmptest <- matrix(0, nrow(xtest), d)
  
  for (j in 1:d) {
    z <- c(x0[, j], x1[, j])
    tmp1 <- tryCatch(gam(y ~ s(z), family = "binomial"),
                     error = function(e) {
                       total.val <- length(unique(z))
                       tryCatch(gam(y ~ s(z, k = total.val), family = "binomial"),
                                error = function(e) NULL)
                     })
    if (!is.null(tmp1)) {
      tmp[, j] <- predict(tmp1, newdata = data.frame(z = z))
      tmptest[, j] <- predict(tmp1, newdata = data.frame(z = xtest[, j]))
    }
  }
  
  list(drtrain = tmp - log(dimorlen(x0)/dimorlen(x1)),
       drtest  = tmptest - log(dimorlen(x0)/dimorlen(x1)))
}

