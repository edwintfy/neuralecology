##############################################################
# m03 compare thinned-model output: negative log likelihood
##############################################################

library(dplyr) ; library(ggplot2)

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
nll %>% 
  mutate(thin = as.integer(thin)) %>% 
  mutate(thin2 = 2^thin) %>% 
  #mutate(thin = factor(thin , levels = sort(unique(as.integer(thin)) , decreasing = TRUE))) %>% 
  arrange(thin) %>%
  ggplot(aes(x = thin , y = nll , color = model)) +
  geom_line() +
  geom_point() +
  facet_grid(~group) +
  scale_color_discrete(labels = c("nn" = "multi-species deep \nneural hierarchical model" , "sn" = "single-species \nneural hierarchical model" , "ss" = "single-species \nbaseline model")) +
  labs(x = "thinning of the training data (log2)" , y = "negative log-likelihood") +
  theme_bw() +
  theme(legend.position = "top")


# ========================================
# graoh with nll and AUC together
# ========================================
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

