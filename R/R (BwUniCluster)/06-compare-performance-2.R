library(tidyverse)
library(pbapply)
library(parallel)
library(assertthat)
library(pROC)
library(patchwork)
source('R/utils-2.R')

theme_set(theme_minimal() + 
            theme(panel.grid.minor = element_blank()))

species <- list.files('out_gru_2', '_nnet.csv', full.names = TRUE) %>%
  gsub('out_gru_2/', '', .) %>%
  gsub('\\_nnet.csv', '', .)

load_ll(species[1])

pboptions(use_lb = TRUE)
cl <- makeCluster(parallel::detectCores())
clusterExport(cl, c('load_ll'))
clusterEvalQ(cl, library(assertthat))
clusterEvalQ(cl, library(tidyverse))
clusterEvalQ(cl, source('R/utils-2.R'))
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
  write_csv('out_round_2/train-valid-nll.csv')

# check for NA values in the NLL values (which result from underflow)
ll_df %>%
  group_by(group) %>%
  summarize(gru_na = sum(gru %>% is.na),
            lstm_na = sum(lstm %>% is.na), 
            vrnn_na = sum(vrnn %>% is.na))


overall_comparisons <- ll_df %>%
  group_by(group) %>%
  summarize(gru_nll = mean(gru, na.rm = TRUE),
            lstm_nll = mean(lstm, na.rm = TRUE), 
            vrnn_nll = mean(vrnn, na.rm = TRUE))
overall_comparisons
write_csv(overall_comparisons, 'out_round_2/nll-comps.csv')

