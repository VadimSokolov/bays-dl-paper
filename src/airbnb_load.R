library(dplyr)
setwd("~/papers/dl-stats/")
sess = readRDS("data/airbnb/sessios.rds")
(dim(distinct(sess, user_id))[1])/(dim(sess)[1])
