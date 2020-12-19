##############################################################
# m01 know the data
##############################################################

library(dplyr) ; library(ggplot2)

# 讀入檔案clean_routes
clean_routes <- read.csv('data/cleaned/clean_routes.csv')
# 快速瀏覽欄位
str(clean_routes)

# 畫出各個route
clean_routes %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(shape = 1)

# route_id中前兩碼代表州
clean_routes %>% 
  tidyr::separate(route_id , c("id1" , "id2") , sep = "_") %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(aes(color = id1) , shape = 1)
# 假如隨機選取部分的州
# 1/2
clean_routes %>% 
  tidyr::separate(route_id , c("id1" , "id2") , sep = "_" , remove = FALSE) %>% 
  dplyr::filter(id1 %in% sample(.$id1 , size = round(length(unique(.$id1)) * 1/2) , replace = FALSE )) %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(aes(color = id1) , shape = 1)
# 3/4
clean_routes %>% 
  tidyr::separate(route_id , c("id1" , "id2") , sep = "_" , remove = FALSE) %>% 
  dplyr::filter(id1 %in% sample(.$id1 , size = round(length(unique(.$id1)) * 3/4) , replace = FALSE )) %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(aes(color = id1) , shape = 1)


# train/ validation/ test
clean_routes %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(aes(color = group) , shape = 1)
# (train, validate, test), blocked by level 3 ecoregion (03_extract route features)
clean_routes %>% 
  xtabs(~group , data = .)
clean_routes %>% 
  tidyr::separate(route_id , c("id1" , "id2") , sep = "_" , remove = FALSE) %>% 
  dplyr::filter(id1 %in% sample(.$id1 , size = round(length(unique(.$id1)) * 1/2) , replace = FALSE )) %>% 
  xtabs(~group , data = .)
clean_routes %>% 
  tidyr::separate(route_id , c("id1" , "id2") , sep = "_" , remove = FALSE) %>% 
  dplyr::filter(id1 %in% sample(.$id1 , size = round(length(unique(.$id1)) * 1/2) , replace = FALSE )) %>% 
  ggplot(aes(x = Longitude , y = Latitude)) +
  geom_point(aes(color = id1) , shape = 1) +
  facet_grid(~group)










