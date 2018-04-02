install.packages("keras")
library(keras)
library(reticulate)
reticulate::use_python("/Users/vsokolov/anaconda2/bin/python")

py_discover_config("keras")
use_condaenv("r-tensorflow")

