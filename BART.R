library(tidymodels)
library(tidyverse)
library(vroom)
library(remotes)
library(parallel)
library(bonsai)
library(lightgbm)
library(dbarts)
library(finetune)


cl <- makePSOCKcluster(8)

MissingTrain <- vroom("./trainWithMissingValues.csv")
train <- vroom("./train.csv")
test <- vroom("./test.csv")

my_recipe <- recipe(formula= type~., data=train) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>% # Convert color to a factor
  step_dummy(color) 

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

folds <- vfold_cv(train, v = 5) 

bart_grid <- grid_regular(
  trees(range = c(50, 500)), # Expanded range for tuning
  levels = 20                 # More levels for finer granularity
)

tune_results <- tune_bayes(
  boost_wf,
  resamples = folds,
  initial = 10, # Number of initial random configurations
  metrics = metric_set(accuracy),
  iter = 30
)

best_params <- select_best(tune_results, metric = "accuracy")

final_workflow <- finalize_workflow(boost_wf, best_params)

final_model <- fit(final_workflow, data = train) 

predictions <- predict(final_model, new_data = test, type = "class")

kaggle_submission <- predictions %>%
  bind_cols(., test) %>%
  dplyr::select(id,.pred_class) %>%
  rename(type = .pred_class)

# Write submission file in required format
vroom_write(kaggle_submission, "./BART.csv", delim = ",")


stopCluster(cl)
