##############################################################
# m04 compare performance
##############################################################

library(dplyr) ; library(tidyr) ; library(readr)
library(pbapply)

# ========================================
# import route information, load species names from the model output files 
# ========================================
# no thinning
#setwd("/Volumes/Li/Neuralecology_out/full training data")
#routes <- read_csv("/Volumes/GoogleDrive/我的雲端硬碟/ich/Documents/2020/【ACA】Reproducibility of Published Statistical Analyses/Neuralecology_VersionControl/Neuralecology_vc/data/cleaned/clean_routes/train-validation by state full/clean_routes.csv") %>% 
#  select(route_id , group)

# 1/2 training data
#setwd("/Volumes/Li/Neuralecology_out/二分之一training data")
#routes <- read_csv("/Volumes/GoogleDrive/我的雲端硬碟/ich/Documents/2020/【ACA】Reproducibility of Published Statistical Analyses/Neuralecology_VersionControl/Neuralecology_vc/data/cleaned/clean_routes/train-validation by state 1_2 train/clean_routes.csv") %>% 
#  select(route_id , group)

# 1/4 training data
setwd("/Volumes/Li/Neuralecology_out/四分之一training data")
routes <- read_csv("/Volumes/GoogleDrive/我的雲端硬碟/ich/Documents/2020/【ACA】Reproducibility of Published Statistical Analyses/Neuralecology_VersionControl/Neuralecology_vc/data/cleaned/clean_routes/train-validation by state 1_4 train/clean_routes.csv") %>% 
  select(route_id , group)

# 1/8 training data
#setwd("/Volumes/Li/Neuralecology_out/八分之一training data")
#routes <- read_csv("/Volumes/GoogleDrive/我的雲端硬碟/ich/Documents/2020/【ACA】Reproducibility of Published Statistical Analyses/Neuralecology_VersionControl/Neuralecology_vc/data/cleaned/clean_routes/train-validation by state 1_16 train/clean_routes.csv") %>% 
#  select(route_id , group)
# 1/16 training data

# 
species <- list.files(path = 'out', pattern = '_ssnet.csv', full.names = TRUE) %>%
  gsub('out/', '', .) %>%
  gsub('\\_ssnet.csv', '', .)

# ========================================
# calculate the predicted occurance probabilities estimated by the three models 
# for each species; each route each year
# ========================================
species_predicted <- function(sp){
  # ssnet ======================================================================
  ssnnet_fit <- file.path('out', paste0(sp, '_ssnet.csv')) %>%
    read_csv %>%
    mutate(method = 'ssnn') %>% 
    full_join(routes , by = "route_id")
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
    full_join(routes , by = "route_id")
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

# ========================================
# calculate for all the species
# save as a list; each species as one data frame
# ========================================
pboptions(use_lb = TRUE)
all.species_predicted <- pblapply(species , 
                                  FUN = function(sp){
                                    species_predicted(sp) %>% 
                                      # only retain necessary columns (to save storage space)
                                      select(sp.bbs , route_id , group , method , year , y , estimate , truth) %>% 
                                      # remove years with no surveys
                                      dplyr::filter(!is.na(y)) %>% 
                                      mutate(method = as.factor(method) , 
                                             year = as.factor(year) , 
                                             group = as.factor(group))
                                  })

# save the output list
save(all.species_predicted , file = "out/all.species_predicted.RData")
#load("out/all.species_predicted.RData")

# ========================================
# calculate the AUC 
# (all the species) (comparison between model types and train/validation set)
# ========================================
AUC_df <- all.species_predicted %>% 
  bind_rows %>% 
  #filter(!is.na(group)) %>% 
  group_by(group , method) %>%               # 比較各個model的train/validation的AUC
  yardstick::roc_auc(truth, estimate) %>% 
  ungroup %>% 
  select(-.metric , -.estimator) %>% 
  rename(AUC = .estimate)

write.csv(AUC_df , file = "out/AUC.csv")

#write.csv(AUC_df , "/Volumes/GoogleDrive/我的雲端硬碟/ich/Documents/2020/【ACA】Reproducibility of Published Statistical Analyses/Neuralecology_VersionControl/Neuralecology_vc/thinning output comparison/AUC_-3.csv")

