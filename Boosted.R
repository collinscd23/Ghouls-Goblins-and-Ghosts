library(tidymodels)
library(tidyverse)
library(vroom)
library(remotes)
library(parallel)
library(bonsai)
library(lightgbm)


cl <- makePSOCKcluster(8)

MissingTrain <- vroom("./trainWithMissingValues.csv")
train <- vroom("./train.csv")
test <- vroom("./test.csv")

my_recipe <- recipe(formula= type~., data=train) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>% # Convert color to a factor
  step_dummy(color) 

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)


folds <- vfold_cv(train, v = 5) 

boost_grid <- grid_regular(tree_depth(range = c(1,3)), trees(range = c(1, 100)),
                           learn_rate(range = c(0, 1)), levels = 10)

tune_results <- tune_grid(
  boost_wf,
  resamples = folds,
  grid = boost_grid,
  metrics = metric_set(accuracy))

best_params <- select_best(tune_results, metric = "accuracy")

final_workflow <- finalize_workflow(boost_wf, best_params)

final_model <- fit(final_workflow, data = train) 

predictions <- predict(final_model, new_data = test, type = "class")

kaggle_submission <- predictions %>%
  bind_cols(., test) %>%
  dplyr::select(id,.pred_class) %>%
  rename(type = .pred_class)

# Write submission file in required format
vroom_write(kaggle_submission, "./BOOSTED.csv", delim = ",")


stopCluster(cl)
