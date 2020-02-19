################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################
# let us create an additional partition of training and test sets from the edx datasset
###############################

set.seed(755, sample.kind = "Rounding") #  in R 3.6 or later
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# to make sure we don't include users and movies in the test set that do not appear in the training set, 
# we remove these entries using the semi_join function 
test_set <- test_set %>%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
  semi_join(train_set, by = 'movieId') %>%
  semi_join(train_set, by = 'userId')

# Define the RMSE as per the Netflix challenge

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Define mu_hat, the average of all ratings

mu_hat <- mean(train_set$rating)
mu_hat

# perform a prediction with all unknown ratings equal to mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# compute movie effect b_i. Drop the hat of mu_hat  

mu <- mean(train_set$rating) 

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) 

# predict with y_hat = mu_hat + b_i 

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

RMSE(test_set$rating, predicted_ratings)

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

# compute user effect b_u 

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# predict with mu + b_i + b_u

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(test_set$rating, predicted_ratings)

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

# check overtraining

# Let’s explore where we made mistakes in our first model, using only movie effects  
# b_i
#  Here are the 10 largest mistakes:

test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10) %>% 
  pull(title)

# First, let’s create a database that connects movieId to movie title:

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

# Here are the 10 best movies according to our estimate:

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10)  %>% 
  pull(title)

#  And here are the 10 worst:

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10)  %>% 
  pull(title) 

# Regularization

# Let’s compute these regularized estimates of b_i using  an optimal λ=  2.25 

lambda <- 2.25
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 


# Do we improve our results?

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

RMSE(predicted_ratings, test_set$rating) 
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))

rmse_results %>% knitr::kable()

# compute RMSE with λ that has been found optimal at 5 for b_i and b_u.

lambda <- 5

mu <- mean(train_set$rating)

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(predicted_ratings, test_set$rating)

model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()

# Date effect 
# refine by adding  d_u, the effect of the date the movie was watched with groupings per week

library("lubridate")

test_set <- test_set  %>% mutate(date = round_date(as_datetime(timestamp), unit = "week"))
train_set <- train_set %>% mutate(date = round_date(as_datetime(timestamp), unit = "week"))

# plot the average rating for each week against date 

train_set %>% group_by(date) %>% 
  summarize(rating = mean(rating)) %>% 
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()


mu <- mean(train_set$rating)

d_u <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(date) %>%
  summarize(d_u = sum(rating - b_i - b_u -mu)/(n() + lambda))

predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(d_u, by ="date") %>%
  mutate(pred = mu + b_i + b_u + d_u) %>%
  pull(pred)

RMSE(predicted_ratings, test_set$rating)

model_5_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie + User + Date Effect Model",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()

# Genres Effect
# plot the error bar plots of the average and standard error for each category of genres
# as there are 796 genres we limit to a sample of 80 genres, or 10%. 

train_set %>% group_by(genres) %>% 
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n()), ) %>%
  filter(n >= 1000) %>%
  sample_n(., 80) %>%                           
  mutate(genres = reorder(genres, avg)) %>% 
  ggplot(aes(x= genres, y = avg, ymin = avg -2*se, ymax = avg + 2*se)) +
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# final computation with optimal  λ =5

lambda <- 5

g_u <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(d_u, by="date") %>%
  group_by(genres) %>%
  summarize(g_u = sum(rating - d_u - b_i - b_u -mu)/(n() + lambda))

predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(d_u, by ="date") %>%
  left_join(g_u, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + d_u + g_u) %>%
  pull(pred)

RMSE(predicted_ratings, test_set$rating)

model_6_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie + User + Date + Genres Effect Model",  
                                     RMSE = model_6_rmse ))
rmse_results %>% knitr::kable()

# Year effect

# investigate effect of year when movie is issued. Extract the year and mutate it.

test_set <- test_set  %>% mutate(year = str_extract(str_extract(test_set$title, "\\(\\d{4}\\)"), "\\d{4}"))
train_set <- train_set  %>% mutate(year = str_extract(str_extract(train_set$title, "\\(\\d{4}\\)"), "\\d{4}"))

# plot the average rating for each year  

train_set %>% group_by(year) %>% 
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>% 
  ggplot(aes(x= year, y = avg, ymin = avg -2*se, ymax = avg + 2*se)) +
  geom_point() +
  geom_errorbar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# compute year effect

y_u <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(d_u, by="date") %>%
  left_join(g_u, by="genres") %>%
  group_by(year) %>%
  summarize(y_u = sum(rating -g_u - d_u - b_i - b_u -mu)/(n() + lambda))

predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(d_u, by ="date") %>%
  left_join(g_u, by = "genres") %>%
  left_join(y_u, by = "year") %>%
  mutate(pred = mu + b_i + b_u + d_u + g_u + y_u) %>%
  pull(pred)

RMSE(predicted_ratings, test_set$rating)

model_7_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie + User + Date + Genres + Year Effect Model",  
                                     RMSE = model_7_rmse ))
rmse_results %>% knitr::kable()

####################################################################################
# let us use edx as training set and validation as test set for the final RMSE scores
####################################################################################

# Define the RMSE as per the Netflix challenge

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Define mu_hat, the average of all ratings

mu_hat <- mean(edx$rating)
mu_hat

# perform a prediction with all unknow ratings equal to mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# compute movie effect b_i. Drop the hat of mu_hat  

mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) 

# predict with y_hat = mu_hat + b_i 

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

RMSE(validation$rating, predicted_ratings)

model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

# compute user effect b_u 

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# predict with y_hat = mu_hat + b_i + b_u

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(validation$rating, predicted_ratings)

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

# Regularization

# Let’s compute these regularized estimates of b_i using  λ=  2.25 

lambda <- 2.25
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 


# Do we improve our results?

predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
RMSE(predicted_ratings, validation$rating) 

model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))

rmse_results %>% knitr::kable()


# compute RMSE with λ optimal at 4.75

lambda <- 4.75

mu <- mean(edx$rating)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(predicted_ratings, validation$rating)

model_4_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()

# refine by adding  d_u with groupings per week 

validation <- validation %>% mutate(date = round_date(as_datetime(timestamp), unit = "week"))
edx <- edx %>% mutate(date = round_date(as_datetime(timestamp), unit = "week"))


d_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(date) %>%
  summarize(d_u = sum(rating - b_i - b_u -mu)/(n() + lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(d_u, by ="date") %>%
  mutate(pred = mu + b_i + b_u + d_u ) %>%
  pull(pred)

RMSE(predicted_ratings, validation$rating)

model_5_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie + User + Date Effect Model",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()

# refine by adding  g_u with genres

g_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(d_u, by="date") %>%
  group_by(genres) %>%
  summarize(g_u = sum(rating - d_u - b_i - b_u -mu)/(n() + lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(d_u, by ="date") %>%
  left_join(g_u, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + d_u + g_u) %>%
  pull(pred)

RMSE(predicted_ratings, validation$rating)

model_6_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie + User + Date + Genres Effect Model",  
                                     RMSE = model_6_rmse ))
rmse_results %>% knitr::kable()

# Year effect


# investigate effect of year when movie is issued. Extract the year and mutate it.

validation <- validation  %>% mutate(year = str_extract(str_extract(validation$title, "\\(\\d{4}\\)"), "\\d{4}"))
edx <- edx  %>% mutate(year = str_extract(str_extract(edx$title, "\\(\\d{4}\\)"), "\\d{4}"))


# compute year effect

y_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(d_u, by="date") %>%
  left_join(g_u, by="genres") %>%
  group_by(year) %>%
  summarize(y_u = sum(rating -g_u - d_u - b_i - b_u -mu)/(n() + lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(d_u, by ="date") %>%
  left_join(g_u, by = "genres") %>%
  left_join(y_u, by = "year") %>%
  mutate(pred = mu + b_i + b_u + d_u + g_u + y_u) %>%
  pull(pred)

RMSE(predicted_ratings, validation$rating)

model_7_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie + User + Date + Genres + Year Effect Model",  
                                     RMSE = model_7_rmse ))
rmse_results %>% knitr::kable()