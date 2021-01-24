library(tidyverse)
#library(ggridges)

# =============== NLL and 100 epochs ===============
out_gru_epoch_100 <- read_csv('out/out_gru_epoch_100.csv', col_names=c('idlist','eplist','train','valid'))
out_lstm_epoch_100 <- read_csv('out/out_lstm_epoch_100.csv', col_names=c('idlist','eplist','train','valid'))
out_rnn_epoch_100 <- read_csv('out/out_rnn_epoch_100.csv', col_names=c('idlist','eplist','train','valid'))

out_gru_epoch_100$type <- 'gru'
out_lstm_epoch_100$type <- 'lstm'
out_rnn_epoch_100$type <- 'rnn'

out_epoch_100 <- rbind(out_gru_epoch_100, out_lstm_epoch_100, out_rnn_epoch_100)

out_epoch_100 %>% pivot_longer(c(train,valid), names_to="data", values_to="nll") %>% 
  ggplot(aes(x=eplist, y=nll, colour=data)) + 
  geom_line() +
  geom_vline(xintercept=5, linetype="dashed") +
  facet_wrap(~type)

out_epoch_100 %>% pivot_longer(c(train,valid), names_to="data", values_to="nll") %>% 
  filter(eplist <= 15) %>% 
  ggplot(aes(x=eplist, y=nll, colour=data)) + 
  geom_line() +
  geom_vline(xintercept=5, linetype="dashed") +
  facet_wrap(~type)




# =============== NLL with 5 epochs ===============
train_valid_nll_1 <- read_csv('out/round_1/train-valid-nll.csv')
train_valid_nll_2 <- read_csv('out/round_2/train-valid-nll.csv')
train_valid_nll_3 <- read_csv('out/round_3/train-valid-nll.csv')
train_valid_nll_4 <- read_csv('out/round_4/train-valid-nll.csv')
train_valid_nll_5 <- read_csv('out/round_5/train-valid-nll.csv')


avgFunc <- function(x1, x2, x3, x4, x5) {
  # make an id column
  x1$id <- 1:nrow(x1)
  x1 <- x1 %>% select(1:4, id, everything())
  x2$id <- 1:nrow(x2)
  x2 <- x2 %>% select(1:4, id, everything())
  x3$id <- 1:nrow(x3)
  x3 <- x3 %>% select(1:4, id, everything())
  x4$id <- 1:nrow(x4)
  x4 <- x4 %>% select(1:4, id, everything())
  x5$id <- 1:nrow(x5)
  x5 <- x5 %>% select(1:4, id, everything())
  
  # merge the 5 datasets by mean
  x <- bind_rows(x1[,5:8], x2[,5:8], x3[,5:8], x4[,5:8], x5[,5:8]) %>%
    group_by(id) %>%   
    summarise_each(funs(mean)) 
  x <- select(x1[,1:5], id, 1:4) %>% left_join(x, by = "id")
  
  return(x)
  }

train_valid_nll <- avgFunc(train_valid_nll_1, train_valid_nll_2, train_valid_nll_3, train_valid_nll_4, train_valid_nll_5)
rm(train_valid_nll_1, train_valid_nll_2, train_valid_nll_3, train_valid_nll_4, train_valid_nll_5)


# train_valid_nll %>% 
#   filter(group=="Training data") %>% 
#   group_by(route_id) %>%
#   summarize(gru_na = sum(gru %>% is.na),
#             lstm_na = sum(lstm %>% is.na), 
#             vrnn_na = sum(vrnn %>% is.na))
# 
# train_valid_nll %>% 
#   filter(group=="Training data") %>% 
#   group_by(route_id) %>% 
#   summarize(gru_nll = mean(gru, na.rm = TRUE),
#             lstm_nll = mean(lstm, na.rm = TRUE), 
#             vrnn_nll = mean(vrnn, na.rm = TRUE))
# 
# train_valid_nll %>% 
#   filter(group=="Validation data") %>% 
#   group_by(route_id) %>% 
#   summarize(gru_nll = mean(gru, na.rm = TRUE),
#             lstm_nll = mean(lstm, na.rm = TRUE), 
#             vrnn_nll = mean(vrnn, na.rm = TRUE))


# check training:validation ratio (~ 7:3)
train_valid_nll %>% 
  filter(group=="Training data") %>% nrow()

train_valid_nll %>% 
  filter(group=="Validation data") %>% nrow()

621742/(1402489 + 621742)


# check total NLL
# training
train_valid_nll %>% 
  filter(group=="Training data") %>% 
  select(gru, lstm, vrnn) %>%
  summarise_all(list(~ sum(.)))

# validation
train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(gru, lstm, vrnn) %>%
  summarise_all(list(~ sum(.)))


# check summary of the 3 models
# training
train_valid_nll %>% 
  filter(group=="Training data") %>% 
  select(gru) %>% summary()

train_valid_nll %>% 
  filter(group=="Training data") %>% 
  select(lstm) %>% summary()

train_valid_nll %>% 
  filter(group=="Training data") %>% 
  select(vrnn) %>% summary()

# validation
train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(gru) %>% summary()

train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(lstm) %>% summary()

train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(vrnn) %>% summary()



# =============== validation NLL in details ===============
# quick check
# 92874 with NLL >= 1.0
train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(gru) %>% filter(gru >= 1.0) %>% count()

# 528868 with NLL < 1.0
train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(gru) %>% filter(gru < 1.0) %>% count()

# 85.1% predictions with NLL < 1.0
528868 / (528868 + 92874)


# 92874 with NLL >= 0.1
train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(gru) %>% filter(gru >= 0.1) %>% count()

# 528868 with NLL < 0.1
train_valid_nll %>% 
  filter(group=="Validation data") %>% 
  select(gru) %>% filter(gru < 0.1) %>% count()

# 61.1% predictions with NLL < 0.1
379983 / (379983 + 241759)



# overall NLL distribution
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data") %>% 
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=50, position='identity') +
  facet_wrap(~type)


# predictions with NLL < 1.0
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll < 1.0) %>% 
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)

# predictions with NLL < 0.1
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll < 0.1) %>% 
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)

# predictions with NLL < 0.05
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll < 0.05) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)

# predictions with NLL < 0.005
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll < 0.005) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)


# predictions with NLL >= 1.0
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll >= 1) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=50, position='identity') +
  facet_wrap(~type)

# predictions with NLL >= 1.0 & nll < 100
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll >= 1 & nll < 100) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=50, position='identity') +
  facet_wrap(~type)

# predictions with NLL >= 1.0 & nll < 25
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll >= 1 & nll < 25) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=50, position='identity') +
  facet_wrap(~type)


# predictions with NLL >= 100
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll >= 100) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=50, position='identity') +
  facet_wrap(~type)

# predictions with NLL >= 200
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll >= 200) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)

# predictions with NLL >= 400
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll >= 400) %>%
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)



# boxplots
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data") %>%
  ggplot(aes(y=nll, x=type)) +
  geom_boxplot()

train_valid_nll %>% 
pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Validation data", nll < 0.3) %>%
  ggplot(aes(y=nll, x=type)) +
  geom_boxplot()



# =============== NLL for training data ===============
# overall NLL distribution for training data
train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Training data") %>% 
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)

train_valid_nll %>% 
  pivot_longer(c(gru, lstm, vrnn), names_to = "type", values_to = "nll") %>% 
  filter(group=="Training data", nll < 1.0) %>% 
  ggplot(aes(x=nll, fill=type)) +
  geom_histogram(bins=70, position='identity') +
  facet_wrap(~type)



# =============== AUC ===============
# gru
auc_df_gru_1 <- read_csv('out/round_1/auc_df-validation_gru.csv')
auc_df_gru_2 <- read_csv('out/round_2/auc_df-validation_gru.csv')
auc_df_gru_3 <- read_csv('out/round_3/auc_df-validation_gru.csv')
auc_df_gru_4 <- read_csv('out/round_4/auc_df-validation_gru.csv')
auc_df_gru_5 <- read_csv('out/round_5/auc_df-validation_gru.csv')

# lstm
auc_df_lstm_1 <- read_csv('out/round_1/auc_df-validation_lstm.csv')
auc_df_lstm_2 <- read_csv('out/round_2/auc_df-validation_lstm.csv')
auc_df_lstm_3 <- read_csv('out/round_3/auc_df-validation_lstm.csv')
auc_df_lstm_4 <- read_csv('out/round_4/auc_df-validation_lstm.csv')
auc_df_lstm_5 <- read_csv('out/round_5/auc_df-validation_lstm.csv')

# rnn
auc_df_rnn_1 <- read_csv('out/round_1/auc_df-validation_rnn.csv')
auc_df_rnn_2 <- read_csv('out/round_2/auc_df-validation_rnn.csv')
auc_df_rnn_3 <- read_csv('out/round_3/auc_df-validation_rnn.csv')
auc_df_rnn_4 <- read_csv('out/round_4/auc_df-validation_rnn.csv')
auc_df_rnn_5 <- read_csv('out/round_5/auc_df-validation_rnn.csv')



avgFunc_auc <- function(x1, x2, x3, x4, x5) {
  # make an id column
  x1$id <- 1:nrow(x1)
  x1 <- x1 %>% select(1:3, id, everything())
  x2$id <- 1:nrow(x2)
  x2 <- x2 %>% select(1:3, id, everything())
  x3$id <- 1:nrow(x3)
  x3 <- x3 %>% select(1:3, id, everything())
  x4$id <- 1:nrow(x4)
  x4 <- x4 %>% select(1:3, id, everything())
  x5$id <- 1:nrow(x5)
  x5 <- x5 %>% select(1:3, id, everything())
  
  # merge the 5 datasets by mean
  x <- bind_rows(x1[,4:5], x2[,4:5], x3[,4:5], x4[,4:5], x5[,4:5]) %>%
    group_by(id) %>%   
    summarise_each(funs(mean)) 
  x <- select(x1[,1:4], id, 1:3) %>% left_join(x, by = "id")
  
  return(x)
}

# gru
auc_df_gru_1$.estimate %>% mean()
auc_df_gru_2$.estimate %>% mean()
auc_df_gru_3$.estimate %>% mean()
auc_df_gru_4$.estimate %>% mean()
auc_df_gru_5$.estimate %>% mean()
auc_df_gru <- avgFunc_auc(auc_df_gru_1, auc_df_gru_2, auc_df_gru_3, auc_df_gru_4, auc_df_gru_5)
auc_df_gru$.estimate %>% mean()

# lstm
auc_df_lstm_1$.estimate %>% mean()
auc_df_lstm_2$.estimate %>% mean()
auc_df_lstm_3$.estimate %>% mean()
auc_df_lstm_4$.estimate %>% mean()
auc_df_lstm_5$.estimate %>% mean()
auc_df_lstm <- avgFunc_auc(auc_df_lstm_1, auc_df_lstm_2, auc_df_lstm_3, auc_df_lstm_4, auc_df_lstm_5)
auc_df_lstm$.estimate %>% mean()

# rnn
auc_df_rnn_1$.estimate %>% mean()
auc_df_rnn_2$.estimate %>% mean()
auc_df_rnn_3$.estimate %>% mean()
auc_df_rnn_4$.estimate %>% mean()
auc_df_rnn_5$.estimate %>% mean()
auc_df_rnn <- avgFunc_auc(auc_df_rnn_1, auc_df_rnn_2, auc_df_rnn_3, auc_df_rnn_4, auc_df_rnn_5)
auc_df_rnn$.estimate %>% mean()


# distributions
auc_df_gru$type <- 'gru'
auc_df_lstm$type <- 'lstm'
auc_df_rnn$type <- 'rnn'

auc_df <- rbind(auc_df_gru, auc_df_lstm, auc_df_rnn)

auc_df %>% 
  ggplot(aes(x=.estimate, fill=type)) + 
  geom_histogram(bins=50, position='identity') +
  labs(x='roc_auc') +
  facet_wrap(~type)
