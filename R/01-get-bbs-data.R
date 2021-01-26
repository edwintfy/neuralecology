##############################################################
# 01 Get data
##############################################################

# *********** dependent libraries ********** #

library(bbsBayes)      # install.packages("requirement/bbsBayes_2.3.4.2020.tgz", repos = NULL, type = .Platform$pkgType)
library(tidyverse)
library(here)

# note this requires mbjoseph/bbsBayes@noninteractive from GitHub
# remotes::install_github("mbjoseph/bbsBayes@noninteractive")


# *********** load data ********** #

# Fetch Breeding Bird Survey dataset (through FTP from USGS database FTP-site)
fetch_bbs_data(level = 'stop') 
# Saving BBS data to /Users/liutzuli/Library/Application Support/bbsBayes

# load these files from the directory where `fetch_bbs_data` saved the data into R
load(list.files("/Users/liutzuli/Library/Application Support/bbsBayes", full.names = TRUE))
str(bbs_data)


# *********** save the data in the working directory `data/bbs_aggregated` ********** #

# ( here() appends its arguments as path components to the root directory )
historical_dir <- here('data', 'bbs_aggregated')
# create a directory (to store the data)
dir.create(historical_dir, recursive = TRUE, showWarnings = FALSE)
# `bbs_data` is a list with three elements (bird, route, species)
# assign the file name to export 
out_files <- historical_dir %>%
  file.path(names(bbs_data)) %>%
  paste0('.csv')
# export the data according to each of the three elements of `bbs_data` 
for (i in seq_along(bbs_data)) {
  write_csv(bbs_data[[i]], out_files[i])
}
# result: three .csv files in `data/bbs_aggregated`


rm(list = ls())

