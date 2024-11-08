library(tidymodels)
library(tidyverse)
library(vroom)
library(remotes)
library(parallel)

cl <- makePSOCKcluster(8)

MissingTrain <- vroom("./trainWithMissingValues.csv")
train <- vroom("./train.csv")
test <- vroom("./test.csv")


my_recipe <- recipe(type~., data = MissingTrain) %>% 
  step_impute_bag(all_numeric_predictors(), impute_with = imp_vars(all_predictors())) 


prep <- prep(my_recipe)
baked <- bake(prep, new_data = MissingTrain)


nn_recipe <- recipe(formula= type~., data=train) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>% # Convert color to a factor
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 250
) %>%
set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

# Define workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

maxHiddenUnits <- 10

# Create tuning grid for hidden units
nn_tuneGrid <- grid_regular(hidden_units(range = c(1, maxHiddenUnits)), levels = 3)

# Tune model
tuned_nn <- nn_wf %>%
  tune_grid(resamples = vfold_cv(train, v = 5), grid = nn_tuneGrid, metrics = metric_set(accuracy))

# Collect and visualize tuning results
tuned_nn %>% collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() +
  labs(title = "Tuning Results for Hidden Units", x = "Hidden Units", y = "Accuracy")

stopCluster(cl)

