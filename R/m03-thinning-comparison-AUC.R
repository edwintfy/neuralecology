##############################################################
# m05 compare thinned-model output: AUC
##############################################################

library(dplyr) ; library(ggplot2)

# ========================================
# 讀入檔案(nll-comps.csv)
# ========================================
AUC_files <- list.files("thinning output comparison" , pattern = "AUC" , full.names = TRUE)

for(i in 1:length(AUC_files)){
  if(i == 1){
    AUC <- read.csv(AUC_files[i]) %>% 
      mutate(thin = strsplit(AUC_files[i] , "_") %>% unlist %>% grep(".csv" , x = . , value = TRUE) %>% gsub(".csv" , "" , x = .)) %>% 
      select(-X)
  }
  AUC <- AUC %>% 
    dplyr::union({
      read.csv(AUC_files[i]) %>% 
        mutate(thin = strsplit(AUC_files[i] , "_") %>% unlist %>% grep(".csv" , x = . , value = TRUE) %>% gsub(".csv" , "" , x = .)) %>% 
        select(-X)
    })
}

# ========================================
# 讀入檔案(nll-comps.csv)
# ========================================
nll_files <- list.files("thinning output comparison" , pattern = "nll" , full.names = TRUE)

for(i in 1:length(nll_files)){
  if(i == 1){
    nll <- read.csv(nll_files[i]) %>% 
      tidyr::pivot_longer(cols = -group , names_to = "model") %>% 
      mutate(thin = strsplit(nll_files[i] , "_") %>% unlist %>% grep(".csv" , x = . , value = TRUE) %>% gsub(".csv" , "" , x = .)) %>% 
      mutate(group = gsub(" data" , "" , group)) %>% 
      mutate(model = gsub("_nll" , "" , model)) %>% 
      rename(nll = value)
  }
  nll <- nll %>% 
    dplyr::union({
      read.csv(nll_files[i]) %>% 
        tidyr::pivot_longer(cols = -group , names_to = "model") %>% 
        mutate(thin = strsplit(nll_files[i] , "_") %>% unlist %>% grep(".csv" , x = . , value = TRUE) %>% gsub(".csv" , "" , x = .)) %>% 
        mutate(group = gsub(" data" , "" , group)) %>% 
        mutate(model = gsub("_nll" , "" , model)) %>% 
        rename(nll = value)
    })
}

# ========================================
# 畫圖
# ========================================
AUC %>% 
  mutate(thin = as.integer(thin)) %>% 
  mutate(thin2 = 2^thin) %>% 
  arrange(thin) %>%
  ggplot(aes(x = thin2 , y = AUC , color = method)) +
  geom_line() +
  geom_point() +
  facet_grid(~group) +
  scale_x_continuous(breaks = 2^(-4:0)) +
  scale_y_continuous(limits = c(0.5,1)) +
  scale_color_discrete(labels = c("nn" = "multi-species deep \nneural hierarchical model" , "ssnn" = "single-species \nneural hierarchical model" , "ss" = "single-species \nbaseline model")) +
  labs(x = "training data size" , y = "overall AUC") +
  theme_bw() +
  theme(legend.position = "top" , 
        axis.text.x = element_text(angle = 90 , vjust = 0.5 , hjust = 1))

AUC %>% 
  mutate(thin = as.integer(thin)) %>% 
  mutate(thin2 = 2^thin) %>% 
  mutate(thin_label = paste0("1/" , 1/thin2)) %>% 
  arrange(thin) %>%
  mutate(method = factor(method , levels = c("nn" , "ssnn" , "ss"))) %>% 
  filter(group == "validation") %>% 
  ggplot(aes(x = thin2 , y = AUC , color = method)) +
  geom_line() +
  geom_point() +
  scale_color_discrete(labels = c("nn" = "multi-species deep \nneural hierarchical model" , "ssnn" = "single-species \nneural hierarchical model" , "ss" = "single-species \nbaseline model")) +
  scale_x_continuous(breaks = 2^(-4:0) , labels = paste0("1/" , 1/2^(-4:0))) +
  scale_y_continuous(limits = c(0.5,1)) +
  labs(x = "training data size (compared to full training data)" , y = "overall AUC") +
  theme_bw() +
  theme(legend.position = "top" , 
        axis.text.x = element_text(angle = 90 , vjust = 0.5 , hjust = 1))

nll %>% 
  mutate(thin = as.integer(thin)) %>% 
  mutate(thin2 = 2^thin) %>% 
  arrange(thin) %>%
  mutate(model = factor(model , levels = c("nn" , "sn" , "ss"))) %>% 
  filter(group == "Validation") %>% 
  ggplot(aes(x = thin2 , y = nll , color = model)) +
  geom_line() +
  geom_point() +
  scale_color_discrete(labels = c("nn" = "multi-species deep \nneural hierarchical model" , "sn" = "single-species \nneural hierarchical model" , "ss" = "single-species \nbaseline model")) +
  scale_x_continuous(breaks = 2^(-4:0) , labels = paste0("1/" , 1/2^(-4:0))) +
  labs(x = "training data size (compared to full training data)" , y = "negative log-likelihood") +
  theme_bw() +
  theme(legend.position = "top" , 
        axis.text.x = element_text(angle = 90 , vjust = 0.5 , hjust = 1))


nll %>% 
  mutate(group = tolower(group) %>% gsub("ing" , "" , .)) %>% 
  mutate(model = ifelse(model == "sn" , "ssnn" , model)) %>% 
  rename(method = model) %>% 
  full_join(AUC , by = c("group" , "method" , "thin")) %>% 
  tidyr::pivot_longer(cols = c(AUC , nll)) %>% 
  mutate(name = gsub("AUC" , "overall AUC" , name) %>% gsub("nll" , "negative log-likelihood" , .)) %>% 
  mutate(thin = as.integer(thin)) %>% 
  mutate(thin2 = 2^thin) %>% 
  arrange(thin) %>%
  mutate(method = factor(method , levels = c("nn" , "ssnn" , "ss"))) %>% 
  filter(group == "validation") %>% 
  ggplot(aes(x = thin2 , y = value , color = method)) +
  geom_line() +
  geom_point() +
  facet_grid(name~. , scales = "free_y") +
  scale_color_discrete(labels = c("nn" = "multi-species \ndeep neural hierarchical model" , "ssnn" = "single-species \nneural hierarchical model" , "ss" = "single-species \nbaseline model")) +
  scale_x_continuous(breaks = 2^(-4:0) , labels = paste0("1/" , 1/2^(-4:0))) +
  labs(x = "training data size (compared to full training data)" , y = "value") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90 , vjust = 0.5 , hjust = 1))


