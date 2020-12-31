##############################################################
# m02 subset the data
##############################################################

library(dplyr) ; library(ggplot2) ; library(sf)

# ========================================
# 讀入檔案clean_routes (original)
# ========================================
clean_routes <- read.csv('data/cleaned/clean_routes/original/clean_routes.csv')

# ========================================
# original train-validation-test set: block by ecoregion
# ========================================
clean_routes %>% 
  mutate(group = factor(group , levels = c("train" , "validation" , "test"))) %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(aes(color = group) , shape = 1) +
  coord_sf(crs = st_crs(4326))

# ========================================
# train-validation set: block by states
# ========================================
set.seed(1221)
clean_routes_group.by.state <-  clean_routes %>% 
  tidyr::separate(route_id , c("state" , "id2") , sep = "_" , remove = FALSE) %>% 
  left_join({
    clean_routes %>% 
      tidyr::separate(route_id , c("state" , "id2") , sep = "_") %>% 
      distinct(state) %>% 
      mutate(group.by.state = sample(c("train" , "validation") , 
                            size = nrow(.) ,
                            replace = TRUE , prob = c(1/2,1/2)))
  } , by = "state") %>% 
  mutate(group = group.by.state)

# plot
clean_routes_group.by.state %>% 
  # re-ordering for visualization
  mutate(group = factor(group , levels = c("train" , "validation"))) %>%
  ggplot(aes(x = Longitude , y = Latitude , color = factor(state))) +
  geom_point(shape = 1) +
  facet_grid(~group) +
  coord_sf(crs = st_crs(4326)) +
  theme(legend.position = "bottom")

# cross table
clean_routes_group.by.state %>% 
  xtabs(~group , data = .)

# ========================================
# thinning
# ========================================
# no thinning: clean_routes_group.by.state
dir.create("data/cleaned/clean_routes/train-validation by state full")
clean_routes_group.by.state %>% 
  write.csv("data/cleaned/clean_routes/train-validation by state full/clean_routes.csv")
# clean_routes_group.by.state <- read.csv("data/cleaned/clean_routes/train-validation by state full/clean_routes.csv")

# thinning- 1/4 routes
set.seed(23150)
clean_routes_thinned.1_4 <- clean_routes_group.by.state %>% 
  filter(group == "train") %>% 
  sample_frac(1/4 , replace = FALSE) %>% 
  dplyr::union({
    clean_routes_group.by.state %>% 
      filter(group == "validation")
  })

clean_routes_thinned.1_4 %>% 
  # re-ordering for visualization
  mutate(group = factor(group , levels = c("train" , "validation"))) %>%
  ggplot(aes(x = Longitude , y = Latitude , color = state)) +
  geom_point(shape = 1) +
  facet_grid(~group) +
  coord_sf(crs = st_crs(4326)) +
  theme(legend.position = "bottom")

dir.create("data/cleaned/clean_routes/train-validation by state 1_4 train")
clean_routes_thinned.1_4 %>% 
  write.csv("data/cleaned/clean_routes/train-validation by state 1_4 train/clean_routes.csv")

# thinning- 1/2 routes
set.seed(23150)
clean_routes_thinned.1_2 <- clean_routes_group.by.state %>% 
  filter(group == "train") %>% 
  sample_frac(1/2 , replace = FALSE) %>% 
  dplyr::union({
    clean_routes_group.by.state %>% 
      filter(group == "validation")
  })

clean_routes_thinned.1_2 %>% 
  # re-ordering for visualization
  mutate(group = factor(group , levels = c("train" , "validation"))) %>%
  ggplot(aes(x = Longitude , y = Latitude , color = factor(state))) +
  geom_point(shape = 1) +
  facet_grid(~group) +
  coord_sf(crs = st_crs(4326)) +
  theme(legend.position = "bottom")

dir.create("data/cleaned/clean_routes/train-validation by state 1_2 train")
clean_routes_thinned.1_2 %>% 
  write.csv("data/cleaned/clean_routes/train-validation by state 1_2 train/clean_routes.csv")


# thinning- 1/8 routes
set.seed(23150)
clean_routes_thinned.1_8 <- clean_routes_group.by.state %>% 
  filter(group == "train") %>% 
  sample_frac(1/8 , replace = FALSE) %>% 
  dplyr::union({
    clean_routes_group.by.state %>% 
      filter(group == "validation")
  })

clean_routes_thinned.1_8 %>% 
  # re-ordering for visualization
  mutate(group = factor(group , levels = c("train" , "validation"))) %>%
  ggplot(aes(x = Longitude , y = Latitude , color = factor(state))) +
  geom_point(shape = 1) +
  facet_grid(~group) +
  coord_sf(crs = st_crs(4326)) +
  theme(legend.position = "bottom")

dir.create("data/cleaned/clean_routes/train-validation by state 1_8 train")
clean_routes_thinned.1_8 %>% 
  write.csv("data/cleaned/clean_routes/train-validation by state 1_8 train/clean_routes.csv")
