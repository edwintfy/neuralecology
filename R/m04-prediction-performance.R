##############################################################
# m04 compare performance
##############################################################

library(dplyr) ; library(tidyr) ; library(readr)
library(pbapply) ; library(parallel)


routes <- read_csv("data/cleaned/clean_routes.csv") %>% 
  select(route_id , group)

species <- list.files(path = 'out', pattern = '_ssnet.csv', full.names = TRUE) %>%
  gsub('out/', '', .) %>%
  gsub('\\_ssnet.csv', '', .)

# calculate the predicted occurance probabilities estimated by the three models 
# for each species; each route each year
species_predicted <- function(sp){
  # ssnet ======================================================================
  ssnnet_fit <- file.path('out', paste0(sp, '_ssnet.csv')) %>%
    read_csv %>%
    mutate(method = 'ssnn') %>% 
    left_join(routes , by = "route_id")
  # the observed (true) occupancy
  ssnnet_observed <- ssnnet_fit %>% 
    select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
    pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")
  # calculate psi
  ssnnet_psi <- ssnnet_fit %>% 
    select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi_") , starts_with("gamma_") , starts_with("psi") ) %>% 
    pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method) ) %>% 
    mutate(name = ifelse(name == "psi0", "psi_1997", name)) %>% 
    separate(name , sep = "_" , into = c("var" , "year") ) %>% 
    pivot_wider(names_from = "var" , values_from = value) %>%
    arrange(sp.bbs, route_id, year) %>%
    mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
    split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
    # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
    lapply(FUN = function(df) {     # # calculate psi from phi and gamma
      for (i in 2:nrow(df)) {
        df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
          (1 - df$psi[i - 1]) * df$gamma[i - 1]
      }
      return(df)
    }) %>%
    bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
    select(-sp_route)
  # predicted  
  ssnnet_predicted <- ssnnet_observed %>% 
    full_join(ssnnet_psi) %>% 
    mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
           estimate = 1 - pr_zero, 
           truth = factor(ifelse(y > 0, 'detected', 'not detected')))
  #
  # nnet  ======================================================================
  # read file
  nnet_fit <- file.path('out', paste0(sp, '_nnet.csv')) %>%
    read_csv %>%
    mutate(method = 'nn') %>% 
    left_join(routes , by = "route_id")
  # the observed (true) occupancy
  nnet_observed <- nnet_fit %>% 
    select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
    pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")
  # calculate psi
  nnet_psi <- nnet_fit %>% 
    select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi_") , starts_with("gamma_") , starts_with("psi") ) %>% 
    pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method) ) %>% 
    mutate(name = ifelse(name == "psi0", "psi_1997", name)) %>% 
    separate(name , sep = "_" , into = c("var" , "year") ) %>% 
    pivot_wider(names_from = "var" , values_from = value) %>%
    arrange(sp.bbs, route_id, year) %>%
    mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
    split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
    # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
    lapply(FUN = function(df) {     # # calculate psi from phi and gamma
      for (i in 2:nrow(df)) {
        df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
          (1 - df$psi[i - 1]) * df$gamma[i - 1]
      }
      return(df)
    }) %>%
    bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
    select(-sp_route)
  # predicted  
  nnet_predicted <- nnet_observed %>% 
    full_join(nnet_psi) %>% 
    mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
           estimate = 1 - pr_zero, 
           truth = factor(ifelse(y > 0, 'detected', 'not detected')))
  #
  # ss ======================================================================
  # read file
  ss_fit <- file.path('out', paste0(sp, '_ss.csv')) %>%
    read_csv %>%
    mutate(method = 'ss') 
  # the observed (true) occupancy
  ss_observed <- ss_fit %>% 
    select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
    pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")
  # calculate psi
  ss_psi <- ss_fit %>% 
    select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi") , starts_with("gamma") , starts_with("psi") ) %>% 
    pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method , -phi , -gamma , -psi1) ) %>% 
    rename(psi = psi1) %>% 
    separate(name , sep = "_" , into = c("var" , "year") ) %>% 
    pivot_wider(names_from = "var" , values_from = value) %>%
    arrange(sp.bbs, route_id, year) %>%
    mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
    split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
    # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
    lapply(FUN = function(df) {     # # calculate psi from phi and gamma
      for (i in 2:nrow(df)) {
        df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
          (1 - df$psi[i - 1]) * df$gamma[i - 1]
      }
      return(df)
    }) %>%
    bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
    select(-sp_route)
  # predicted  
  ss_predicted <- ss_observed %>% 
    full_join(ss_psi) %>% 
    mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
           estimate = 1 - pr_zero, 
           truth = factor(ifelse(y > 0, 'detected', 'not detected')))
  # 
  # combine, return ======================================================================
  rm(ssnnet_fit , ssnnet_observed , ssnnet_psi , nnet_fit , nnet_observed , nnet_psi , ss_fit , ss_observed , ss_psi)
  predicted <- ssnnet_predicted %>% 
    bind_rows(nnet_predicted) %>% 
    bind_rows(ss_predicted)
  return(predicted)
}

# one example: 
species_predicted(species[1])

# calculate for all the species to calculate AUC
pboptions(use_lb = TRUE)
all.species_predicted <- pblapply(species , 
                                  FUN = function(sp){
                                    species_predicted(sp) %>% 
                                      # only retain necessary columns
                                      select(sp.bbs , route_id , group , method , year , y , estimate , truth) %>% 
                                      # remove years with no surveys
                                      dplyr::filter(!is.na(y))
                                  })
#save(all.species_predicted , file = "out/all.species_predicted.RData")

all.species_predicted %>% 
  bind_rows %>% 
  group_by(group , method) %>%               # 比較各個model的train/validation的AUC
  yardstick::roc_auc(truth, estimate) %>% 
  ungroup %>% 
  select(-.metric , -.estimator) %>% 
  rename(AUC = .estimate)




###

species_route_AUC <- function(sp){
  # ssnet的AUC  ==========================================================================
  {
    # read file
    ssnnet_fit <- file.path('out', paste0(sp, '_ssnet.csv')) %>%
      read_csv %>%
      mutate(method = 'ssnn') %>% 
      left_join(routes , by = "route_id")
    # the observed (true) occupancy
    ssnnet_observed <- ssnnet_fit %>% 
      select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
      pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")
    # calculate psi
    ssnnet_psi <- ssnnet_fit %>% 
      select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi_") , starts_with("gamma_") , starts_with("psi") ) %>% 
      pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method) ) %>% 
      mutate(name = ifelse(name == "psi0", "psi_1997", name)) %>% 
      separate(name , sep = "_" , into = c("var" , "year") ) %>% 
      pivot_wider(names_from = "var" , values_from = value) %>%
      arrange(sp.bbs, route_id, year) %>%
      mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
      split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
      # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
      lapply(FUN = function(df) {     # # calculate psi from phi and gamma
        for (i in 2:nrow(df)) {
          df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
            (1 - df$psi[i - 1]) * df$gamma[i - 1]
        }
        return(df)
      }) %>%
      bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
      select(-sp_route)
    # predicted  
    ssnnet_predicted <- ssnnet_observed %>% 
      full_join(ssnnet_psi) %>% 
      mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
             estimate = 1 - pr_zero, 
             truth = factor(ifelse(y > 0, 'detected', 'not detected')))
    ssnnet_auc <- ssnnet_predicted %>% 
      # remove years with no surveys
      filter(!is.na(y)) %>% 
      group_by(route_id , group) %>%               # 比較各個routes之間的prediction performance
      yardstick::roc_auc(truth, estimate) %>% 
      ungroup %>% 
      select(-.metric , -.estimator) %>% 
      rename(AUC = .estimate) %>% 
      mutate(sp.bbs = unique(ssnnet_predicted$sp.bbs)) %>% 
      mutate(method = unique(ssnnet_predicted$method))
  }
  
  # nnet的AUC ==========================================================================
  {
    # read file
    nnet_fit <- file.path('out', paste0(sp, '_nnet.csv')) %>%
      read_csv %>%
      mutate(method = 'nn') %>% 
      left_join(routes , by = "route_id")
    # the observed (true) occupancy
    nnet_observed <- nnet_fit %>% 
      select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
      pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")
    # calculate psi
    nnet_psi <- nnet_fit %>% 
      select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi_") , starts_with("gamma_") , starts_with("psi") ) %>% 
      pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method) ) %>% 
      mutate(name = ifelse(name == "psi0", "psi_1997", name)) %>% 
      separate(name , sep = "_" , into = c("var" , "year") ) %>% 
      pivot_wider(names_from = "var" , values_from = value) %>%
      arrange(sp.bbs, route_id, year) %>%
      mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
      split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
      # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
      lapply(FUN = function(df) {     # # calculate psi from phi and gamma
        for (i in 2:nrow(df)) {
          df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
            (1 - df$psi[i - 1]) * df$gamma[i - 1]
        }
        return(df)
      }) %>%
      bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
      select(-sp_route)
    # predicted  
    nnet_predicted <- nnet_observed %>% 
      full_join(nnet_psi) %>% 
      mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
             estimate = 1 - pr_zero, 
             truth = factor(ifelse(y > 0, 'detected', 'not detected')))
    nnet_auc <- nnet_predicted %>% 
      # remove years with no surveys
      filter(!is.na(y)) %>% 
      group_by(route_id , group) %>%               # 比較各個routes之間的prediction performance
      yardstick::roc_auc(truth, estimate) %>% 
      ungroup %>% 
      select(-.metric , -.estimator) %>% 
      rename(AUC = .estimate) %>% 
      mutate(sp.bbs = unique(nnet_predicted$sp.bbs)) %>% 
      mutate(method = unique(nnet_predicted$method))
  }
  
  # ss的AUC ==========================================================================
  {
    # read file
    ss_fit <- file.path('out', paste0(sp, '_ss.csv')) %>%
      read_csv %>%
      mutate(method = 'ss') 
    # the observed (true) occupancy
    ss_observed <- ss_fit %>% 
      select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
      pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")
    # calculate psi
    ss_psi <- ss_fit %>% 
      select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi") , starts_with("gamma") , starts_with("psi") ) %>% 
      pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method , -phi , -gamma , -psi1) ) %>% 
      rename(psi = psi1) %>% 
      separate(name , sep = "_" , into = c("var" , "year") ) %>% 
      pivot_wider(names_from = "var" , values_from = value) %>%
      arrange(sp.bbs, route_id, year) %>%
      mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
      split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
      # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
      lapply(FUN = function(df) {     # # calculate psi from phi and gamma
        for (i in 2:nrow(df)) {
          df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
            (1 - df$psi[i - 1]) * df$gamma[i - 1]
        }
        return(df)
      }) %>%
      bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
      select(-sp_route)
    # predicted  
    ss_predicted <- ss_observed %>% 
      full_join(ss_psi) %>% 
      mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
             estimate = 1 - pr_zero, 
             truth = factor(ifelse(y > 0, 'detected', 'not detected')))
    ss_auc <- ss_predicted %>% 
      # remove years with no surveys
      filter(!is.na(y)) %>% 
      group_by(route_id , group) %>%               # 比較各個routes之間的prediction performance
      yardstick::roc_auc(truth, estimate) %>% 
      ungroup %>% 
      select(-.metric , -.estimator) %>% 
      rename(AUC = .estimate) %>% 
      mutate(sp.bbs = unique(ss_predicted$sp.bbs)) %>% 
      mutate(method = unique(ss_predicted$method))
  }
  
  # combine the AUC from the three models  ==========================================================================
  rm(ssnnet_fit , ssnnet_observed , ssnnet_predicted , ssnnet_psi , 
     nnet_fit , nnet_observed , nnet_predicted , nnet_psi ,
     ss_fit , ss_observed , ss_predicted , ss_psi)
  sp.auc <- ssnnet_auc %>% 
    bind_rows(nnet_auc) %>% 
    bind_rows(ss_auc) %>% 
    pivot_wider(names_from = method , values_from = AUC)
  
  # write the AUC of the species   ==========================================================================
  if(!dir.exists("out/AUC")) dir.create("out/AUC")
  write_csv(sp.auc , 
            paste0("out/AUC/" , sp , "_AUC.csv" ))
}

species_route_AUC(sp = species[2])












#####

{
  ssnnet_fit <- file.path('out', paste0(sp, '_ssnet.csv')) %>%
    read_csv %>%
    mutate(method = 'ssnn') %>% 
    left_join(routes , by = "route_id")
  
  nnet_fit <- file.path('out', paste0(sp, '_nnet.csv')) %>%
    read_csv %>%
    mutate(method = 'nn') %>% 
    left_join(routes , by = "route_id")
  
  
  ss_fit <- file.path('out', paste0(sp, '_ss.csv')) %>%
    read_csv %>%
    mutate(method = 'ss') 
}

# the observed (true) occupancy
observed <- ssnnet_fit %>% 
  select(sp.bbs , route_id , group , method , as.character(1997:2019) ) %>% 
  pivot_longer(cols = as.character(1997:2019) , values_to = "y" , names_to = "year")

# calculate psi
psi <- ssnnet_fit %>% 
  select(sp.bbs , route_id , group , method , starts_with("p_") , starts_with("phi_") , starts_with("gamma_") , starts_with("psi") ) %>% 
  pivot_longer(cols = c(-sp.bbs , -route_id , -group , -method) ) %>% 
  mutate(name = ifelse(name == "psi0", "psi_1997", name)) %>% 
  separate(name , sep = "_" , into = c("var" , "year") ) %>% 
  pivot_wider(names_from = "var" , values_from = value) %>%
  arrange(sp.bbs, route_id, year) %>%
  mutate(sp_route = paste(sp.bbs, route_id, sep = "__")) %>%
  split(.$sp_route) %>%           # 把每一個species-route 組合切成獨立的data.frame
  # 針對每一個species-route的data.frame計算psi 得到多一個column為psi
  lapply(FUN = function(df) {     # # calculate psi from phi and gamma
    for (i in 2:nrow(df)) {
      df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
        (1 - df$psi[i - 1]) * df$gamma[i - 1]
    }
    return(df)
  }) %>%
  bind_rows() %>%                 # 再把所有species-route的data.frame計算出來的psi合併
  select(-sp_route)

# predicted  
predicted <- observed %>% 
  full_join(psi) %>% 
  mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
         estimate = 1 - pr_zero, 
         truth = factor(ifelse(y > 0, 'detected', 'not detected')))

{ # cross table: cut point = 0.5
  predicted_xtab <- predicted %>% 
    mutate(truth = factor(truth , levels = c("not detected" , "detected"))) %>% 
    mutate(etd.occupancy = ifelse(estimate >= 0.5 , "1" , "0")) %>% 
    # remove years with no surveys
    filter(!is.na(y)) %>% 
    # validation set
    filter(group == "validation") %>% 
    xtabs(~truth+etd.occupancy , data = .) %>% 
    as.matrix() 
  
  sum(diag(predicted_xtab)) / sum(predicted_xtab)  # equal weight 
}


predicted_auc <- predicted %>% 
  # remove years with no surveys
  filter(!is.na(y)) %>% 
  # validation set
  filter(group == "validation") %>% 
  group_by(route_id , group) %>%               # 比較各個routes之間的prediction performance
  yardstick::roc_auc(truth, estimate) %>% 
  ungroup %>% 
  select(-.metric , -.estimator) %>% 
  rename(AUC = .estimate) %>% 
  mutate(sp.bbs = unique(predicted$sp.bbs)) %>% 
  mutate(method = unique(predicted$method))


