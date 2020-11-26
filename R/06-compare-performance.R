##############################################################
# 06 compare performance
##############################################################
library(tidyverse)
library(pbapply)
library(parallel)
library(assertthat)
library(pROC)
library(patchwork)         # install.packages("patchwork")
source('R/utils.R')

theme_set(theme_minimal() + 
            theme(panel.grid.minor = element_blank()))

# 得到的output是每個物種一個csv檔 存在out資料夾
# 讀入這些csv檔的名字
species <- list.files(path = 'out', pattern = '_ssnet.csv', full.names = TRUE) %>%
  gsub('out/', '', .) %>%
  gsub('\\_ssnet.csv', '', .)

load_ll(species[1])

pboptions(use_lb = TRUE)
# Creates a set of copies of R running in parallel and communicating over sockets.
cl <- makeCluster(parallel::detectCores())       # parallel::detectCores()... 4 cores on my mac
clusterExport(cl, c('load_ll'))
# Apply Operations using Clusters; These functions provide several ways to parallelize computations using a cluster.
clusterEvalQ(cl, library(assertthat))
clusterEvalQ(cl, library(tidyverse))
clusterEvalQ(cl, source('R/utils.R'))
# Adding Progress Bar to '*apply' Functions
ll_dfs <- pblapply(species, load_ll, cl = cl)
stopCluster(cl)


ll_df <- ll_dfs %>%
  lapply(function(x) x$nll) %>%
  bind_rows %>%
  filter(group %in% c('train', 'validation')) %>%
  mutate(group = ifelse(group == 'train', 
                        'Training data', 
                        'Validation data'))

ll_df %>%
  write_csv('out/train-valid-nll.csv')

# check for NA values in the NLL values (which result from underflow)
ll_df %>%
  group_by(group) %>%
  summarize(nn_na = sum(nn %>% is.na),
            ss_na = sum(ss %>% is.na), 
            sn_na = sum(ssnn %>% is.na))


overall_comparisons <- ll_df %>%
  group_by(group) %>%
  summarize(nn_nll = mean(nn, na.rm = TRUE),
            ss_nll = mean(ss, na.rm = TRUE), 
            sn_nll = mean(ssnn, na.rm = TRUE))
overall_comparisons
write_csv(overall_comparisons, 'out/nll-comps.csv')
