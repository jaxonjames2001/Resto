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

pen_model <- linear_reg(mixture=tune(), 
                        penalty=tune()) %>%
  set_engine("glmnet")

penalized_workflow <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%
  add_model(pen_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- penalized_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(rmse))

bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <- penalized_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

penalized_predictions <- final_wf %>%
  predict(new_data = test, type = "numeric")

submission <- penalized_predictions %>%
  bind_cols(., test) %>%
  select(Id, .pred) %>%
  rename(Prediction = .pred)

vroom_write(x=submission, file="./PenPreds.csv", delim=",")