load_ll <- function(sp) {
  
  gru_fit <- file.path('out_gru_2', paste0(sp, '_nnet.csv')) %>%
    read_csv %>%
    mutate(method = 'gru')
  
  lstm_fit <- file.path('out_lstm_2', paste0(sp, '_nnet.csv')) %>%
    read_csv %>%
    mutate(method = 'lstm')
  
  vrnn_fit <- file.path('out_rnn_2', paste0(sp, '_nnet.csv')) %>%
    read_csv %>%
    mutate(method = 'vrnn')
  
  ss_fit <- file.path('out_ss', paste0(sp, '_ss.csv')) %>%
    read_csv %>%
    mutate(method = 'ss')
  
  gru_fit$loglik <- gru_fit %>%
    split(sort(as.numeric(rownames(.)))) %>% 
    lapply(neuralnet_ll) %>%
    unlist
  
  lstm_fit$loglik <- lstm_fit %>%
    split(sort(as.numeric(rownames(.)))) %>% 
    lapply(neuralnet_ll) %>%
    unlist
  
  vrnn_fit$loglik <- vrnn_fit %>%
    split(sort(as.numeric(rownames(.)))) %>% 
    lapply(neuralnet_ll) %>%
    unlist
  
  joined <- gru_fit %>%
    full_join(lstm_fit) %>%
    full_join(vrnn_fit)
  
  nll <- joined %>%
    group_by(method, sp.bbs, route_id) %>%
    summarize(nll = -sum(loglik)) %>%
    ungroup %>%
    left_join(distinct(ss_fit, sp.bbs, route_id, group, english)) %>%
    spread(method, nll) %>%
    filter(group != 'test')
  
  list(gru = gru_fit, 
       lstm = lstm_fit, 
       vrnn = vrnn_fit,
       nll = nll)
}



# Forward algorithm helper functions --------------------------------------

# get probabilities of observations as diagonal matrix
get_y_prob <- function(y, p, k = 50) {
  if (!is.na(y)) {
    prob <- diag(c(dbinom(y, size = k, prob = p), as.numeric(y == 0)))
  } else {
    prob <- diag(2)
  }
  prob
}



# Compute log-likelihood of neural network with time varying transitions
neuralnet_ll <- function(df_row) {
  y <- select(df_row, as.character(1997:2018)) %>%
    unlist
  p <- select(df_row, starts_with('p_')) %>%
    unlist
  phi <- select(df_row, starts_with('phi_')) %>%
    unlist
  gamma <- select(df_row, starts_with('gamma_')) %>%
    unlist
  
  nyear <- length(y)
  
  c_t <- rep(NA, nyear)
  prods_raw <- matrix(c(df_row$psi0, 1 - df_row$psi0), nrow = 1, ncol = 2) %*%
    get_y_prob(y = y[1], p = p[1])
  c_t[1] <- 1 / sum(prods_raw)
  prods <- c_t[1] * prods_raw
  
  for (t in 1:(nyear - 1)) {
    prods_raw <- prods %*% 
      matrix(c(phi[t], gamma[t], 1 - phi[t], 1 - gamma[t]), 
             nrow = 2, ncol = 2) %*%
      get_y_prob(y = y[t + 1], p = p[t + 1])
    c_t[t + 1] <- 1 / sum(prods_raw)
    prods <- c_t[t + 1] * prods_raw
  }
  log_lik <- -sum(log(c_t))
  log_lik
}



# Compute log likelihood of baseline model
# Note: this is slightly different than the full model, because the transition
# probabilities are time-invariant
# baseline_ll <- function(df_row) {
#   y <- select(df_row, as.character(1997:2018)) %>%
#     unlist
#   p <- select(df_row, starts_with('p_')) %>%
#     unlist
#   nyear <- length(y)
#   Omega <- matrix(c(df_row$phi, df_row$gamma, 1 - df_row$phi, 1 - df_row$gamma), 
#                   nrow = 2, ncol = 2)
#   c_t <- rep(NA, nyear)
#   prods_raw <- matrix(c(df_row$psi1, 1 - df_row$psi1), nrow = 1, ncol = 2) %*%
#     get_y_prob(y = y[1], p = p[1])
#   c_t[1] <- 1 / sum(prods_raw)
#   prods <- c_t[1] * prods_raw
#   
#   for (t in 1:(nyear - 1)) {
#     prods_raw <- prods %*% Omega %*% get_y_prob(y = y[t + 1], p = p[t + 1])
#     c_t[t + 1] <- 1 / sum(prods_raw)
#     prods <- c_t[t + 1] * prods_raw
#   }
#   log_lik <- -sum(log(c_t))
#   log_lik
# }