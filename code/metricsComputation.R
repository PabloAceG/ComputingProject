# Libraries
library(ECoL)

#data("iris")

# Load the dataset to be used
irisorio <- read.table("C:/Users/Pablo/Documents/Git/ComputingProject/dataset/iris.csv",
                     header = TRUE,
                     sep = ",")

#lastColNumber <- ncol(apache)
#astRowNumber <- nrow(apache)


# Data related to the Apache code.
#bugsApacheData <- apache[, 3 : lastColNumber - 1 ]
# Number of faults.
#numFaultsApache <-apache[, lastColNumber ]

iris_input <- irisorio[, 1 : 4]
iris_target <- irisorio[, 5]

# Measures
overlapping(iris_input, iris_target)
