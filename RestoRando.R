library(lubridate)
library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(recipes)
library(embed)
library(lme4)
library(kknn)
library(dplyr)
library(recipes)

test <- vroom("test.csv")
train <- vroom("train.csv")

recipeTargetEncoded <- recipe(revenue ~ ., data = train) %>%
  step_rename(Open_date = `Open Date`,
              City_group = `City Group`) %>%
  step_mutate(Open_date = as.Date(Open_date, format = "%m/%d/%Y"),
              City = factor(City),
              City_group = factor(City_group),
              Type = factor(Type),
              Id = factor(Id)) %>%
  step_normalize(all_numeric_predictors()) %>%
  update_role(Id, new_role = "ID") %>%
  step_date(Open_date, features = c("dow", "month", "year", "quarter")) %>%
  step_rm(Id, Open_date) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors())

# Define model
tree_mod <- rand_forest(mtry = tune(), 
                        min_n = tune(), 
                        trees = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression") 

# Create workflow
tree_wf <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%  # Ensure the recipe is properly defined
  add_model(tree_mod)

# Set up grid of tuning values
tuning_params <- grid_regular(
  mtry(range = c(1, 10)),  # Specify range for mtry
  min_n(range = c(2, 10)),  # Specify range for min_n
  trees(range = c(100, 1000)),  # Specify range for trees
  levels = 5  # Adjust levels as needed
)

# Set up k-fold cross-validation
folds <- vfold_cv(train, v = 5, repeats = 1)

# Perform tuning
CV_results <- tree_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_params,
    metrics = metric_set(rmse)  # RMSE is appropriate for regression
  )

# Find best tuning parameters based on RMSE
bestTune <- CV_results %>%
  select_best(metric = "rmse")

# Finalize workflow with the best tuning parameters and fit the model
final_tree_wf <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)  # Fit on the training data


RAND_predictions <- final_tree_wf %>%
  predict(new_data = test, type = "numeric")

submission <- RAND_predictions %>%
  bind_cols(., test) %>%
  select(Id, .pred) %>%
  rename(Prediction = .pred)

vroom_write(x=submission, file="./RANDPreds.csv", delim=",")

