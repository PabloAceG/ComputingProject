# Libraries
library(ECoL)

# Load the dataset to be used
apache <- read.table("C:/Users/Pablo/Documents/Git/DASE/datasets/defectPred/BugCatchers/Apache.csv",
                     header = TRUE,
                     sep = ",",
                     row.names = 1)

lastColNumber <- ncol(apache)
lastRowNumber <- nrow(apache)


# Data related to the Apache code.
bugsApacheData <- apache[, 3 : lastColNumber - 1 ]
# Number of faults.
numFaultsApache <-apache[, lastColNumber ]


# Measures
overlapping(bugsApacheData, numFaultsApache)
