# November 30, 2025
# File to make the t and Gaussian simulation plots

################################################################################
# Load libraries
library(reshape2)
library(ggplot2)

################################################################################
# Read in data

t5 <- readRDS(here("Simulation/simulation data/p32-t-ionosphere-df5-m990.RDS"))
t10 <- readRDS(here("Simulation/simulation data/p32-t-ionosphere-df10-m990.RDS"))
t30 <- readRDS(here("Simulation/simulation data/p32-t-ionosphere-df30-m990.RDS"))
gauss <- readRDS(here("Simulation/simulation data/from120-32-diff-cov-mean-m990.RDS"))

################################################################################

# Define function to put the data in long format
make_long <- function(mat, label) {
  df <- as.data.frame(mat[,1:3])
  df$index <- seq(from = 120, to = 990, by = 30)
  
  df_long <- melt(df, id.vars = "index",
                  variable.name = "line",
                  value.name   = "value")
  
  df_long$line <- factor(df_long$line,
                         levels = c("V1","V2","V3"),
                         labels = c("Logistic Regression",
                                    "Naive Bayes",
                                    "Smart Bayes"))
  
  df_long$dist <- label
  df_long
}

# Put the data in long format and bind the rows
df_all <- bind_rows(
  make_long(gauss, "Gaussian"),
  make_long(t5,    "Multivariate t, df = 5"),
  make_long(t10,   "Multivariate t, df = 10"),
  make_long(t30,   "Multivariate t, df = 30")
)

# Set the factors appropriately
df_all$dist <- factor(df_all$dist,
                      levels = c("Multivariate t, df = 5",
                                 "Multivariate t, df = 10", 
                                 "Multivariate t, df = 30", "Gaussian"))

#----------------------------------------------------------
# Faceted plot (one legend, one title)
#----------------------------------------------------------
# Generate the PDF
pdf(
  file =
    paste0(
      here("Simulation/plots/"),
      "faceted-plot-four-simulations-small-lines.pdf"
    )
)

p <- ggplot(df_all, aes(x = index, y = value, color = line)) +
  geom_line(size = .5) +
  scale_color_manual(values = c("orange", "blue", "black")) +
  scale_x_continuous(limits = c(120, 990),
                     breaks = seq(120, 990, by = 60)) +
  labs(
    x = "Training size",
    y = "Misclassification rate",
    color = "Legend"
  ) +
  facet_wrap(~ dist, ncol = 1) +   # vertical layout (change ncol as you like)
  ggtitle("Misclassification rate vs. Training size for Gaussian and t simulations") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom"     # keeps one unified legend
  )

p
dev.off()
