library(tidymodels)
library(tidyverse)
library(vroom)


cl <- makePSOCKcluster(8)

MissingTrain <- vroom("./trainWithMissingValues.csv")
train <- vroom("./train.csv")
test <- vroom("./test.csv")


my_recipe <- recipe(type~., data = MissingTrain) %>% 
  step_impute_bag(all_numeric_predictors(), impute_with = imp_vars(all_predictors())) 


prep <- prep(my_recipe)
baked <- bake(prep, new_data = MissingTrain)



stopCluster(cl)