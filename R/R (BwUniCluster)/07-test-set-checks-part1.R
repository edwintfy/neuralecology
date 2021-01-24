library(tidyverse)
library(pbapply)
library(parallel)
library(assertthat)
#library(sf)
library(pROC)
library(patchwork)
library(vroom)
library(yardstick)
#library(rmapshaper)
library(here)
source('R/utils.R')


species <- list.files('out', '_finalnet.csv', full.names = TRUE) %>%
  gsub('out/', '', .) %>%
  gsub('\\_finalnet.csv', '', .)

dir.create(here::here('out', 'q_dfs'), showWarnings = FALSE, recursive = TRUE)

load_final_fit <- function(sp) {
  routes <- read_csv('data/cleaned/clean_routes.csv')
  
  nnet_fit <- here::here('out', paste0(sp, '_finalnet.csv')) %>%
    read_csv %>%
    mutate(method = 'nn') %>%
    left_join(select(routes, route_id, group))
  
  y_obs <- nnet_fit %>%
    select(sp.bbs, route_id, group, as.character(1997:2017)) %>%
    gather(var, value, -sp.bbs, -route_id, -group) %>%
    mutate(var = ifelse(!grepl("_", var), 
                        paste('y', var, sep = "_"), 
                        var)) %>%
    separate(var, into = c('var', 'year'), sep = '_') %>%
    rename(y = value) %>%
    select(-var) %>%
    mutate(sp.bbs = as.character(sp.bbs))
  
  calc_psi <- function(df) {
    for (i in 2:nrow(df)) {
      df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
        (1 - df$psi[i - 1]) * df$gamma[i - 1]
    }
    df
  }
  
  psi_df <- nnet_fit %>%
    select(sp.bbs, route_id, group, starts_with("p_"), starts_with("phi_"), 
           starts_with("gamma_"), starts_with("psi")) %>%
    gather(var, value, -sp.bbs, -route_id, -group) %>%
    mutate(var = ifelse(var == "psi0", "psi_1997", var)) %>%
    separate(var, into = c("var", "year")) %>%
    spread(var, value) %>%
    arrange(sp.bbs, route_id, year) %>%
    unite("sp_route", sp.bbs, route_id, sep = "__") %>%
    split(.$sp_route) %>%
    lapply(FUN = calc_psi) %>%
    bind_rows() %>%
    separate(sp_route, into = c('sp.bbs', 'route_id'), sep = "__")
  
  quant_df <- psi_df %>%
    mutate(marginal_pred = psi * qbinom(.5, size = 50, prob = p),
           conditional_pred = ifelse(psi > .5, 
                                     qbinom(.5, size = 50, prob = p), 
                                     0), 
           conditional_lo = ifelse(psi > .5, 
                                   qbinom(.025, size = 50, prob = p), 
                                   0), 
           conditional_hi = ifelse(psi > .5, 
                                   qbinom(.975, size = 50, prob = p), 
                                   0), 
           phi = phi, 
           gamma = gamma) %>%
    left_join(y_obs)
  
  out_name <- here::here('out', 'q_dfs', paste0(quant_df$sp.bbs[1], '.csv'))
  write_csv(quant_df, out_name)
  
  quant_df
}

test <- load_final_fit(species[1])

# write out q_dfs for each species
dir.create(here::here('out', 'q_dfs'), showWarnings = FALSE)
pboptions(use_lb = TRUE)
cl <- makeCluster(parallel::detectCores())
clusterEvalQ(cl, library(tidyverse))
ll_dfs <- pblapply(species, load_final_fit, cl = cl)
stopCluster(cl)

save(ll_dfs, file="ll_dfs.RData")