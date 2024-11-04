library(tidymodels)
library(tidyverse)
library(vroom)


cl <- makePSOCKcluster(8)

MissingTrain <- vroom("./trainWithMissingValues.csv")
train <- vroom("./train.csv")
test <- vroom("./test.csv")

my_recipe <- recipe(type~., data = train) %>% 
  step_impute_mean(all_numeric_predictors()) 


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

stopCluster(cl)