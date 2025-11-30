# November 30, 2025
# File to make plots of experiment results
# Specifically to make plots of AWS runs

################################################################################
library(stringr)

# Find the RDS files we want to loop through
rds.files <- list.files(path = here("Real Data/aws-results/"),
                        pattern = "200.RDS")

# Loop through the RDS files
for(rds in rds.files){
  
  # Read in the file
  a <- readRDS(rds)
  
  # Get the plot range (max and min)
  plot.range <- range(a[c("glm_mean","nb_mean","sb_mean")])
  
  # Grab the data name
  data.name <- str_match(rds, pattern = "-(.*?)-")[2]
  
  # Make a PDF file for the plot
  # Change this location if a different one is preferred
  pdf(file = paste0(here("Real Data/plots/"),data.name,"-lr.pdf"))
  
  # Make an empty plot
  plot(x = a$m, 
       y = seq(from = max(0, plot.range[1]-.01), 
               to = min(0.5, plot.range[2]+.01),
               length = length(a$m)), 
       type = "n",
       xlab = "Size of training data", ylab = "Misclassification rate")
  
  # Put it in the title
  title(paste0("Misclassification rates: ", data.name))
  
  # Add lines to the plot
  lines(x = a$m, y = a$glm_mean, col = "orange")
  lines(x = a$m, y = a$nb_mean, col = "blue")
  lines(x = a$m, y = a$dab_mean, col = "black")
  
  # Add a legend
  legend("topright", legend = c("LR", "NB","SB"), 
         col = c("orange","blue","black"), lty = c(1,1,1),
         bg = "white")
  
  # Close the PDF
  dev.off()
  
}