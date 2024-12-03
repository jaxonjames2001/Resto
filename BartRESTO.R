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

# define model
bart_mod <- bart(mode = "regression", trees=100) %>%
  set_engine("dbarts")

# create workflow
bart_wf <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%
  add_model(bart_mod) %>%
  fit(data=train)

bart_preds <- predict(bart_wf, new_data=test,type="numeric")

submission <- bart_preds %>%
  bind_cols(., test) %>%
  select(Id, .pred) %>%
  rename(Prediction = .pred)

vroom_write(x=submission, file="./BARTPreds.csv", delim=",")
