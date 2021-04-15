# Imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

# Pretty print cross-validation and training loss
def print_model_report(
    model,
    predictors,
    target,
    scoring="neg_root_mean_squared_error",
    cv=True,
    cv_folds=5,
):
    _ = model.fit(predictors, target)
    if cv:
        cv_score = -cross_val_score(
            model, predictors, target, cv=cv_folds, scoring=scoring
        ).mean()
        print("CV loss:", cv_score)
    preds = model.predict(predictors)
    training_score = mean_squared_error(preds, target, squared=False)
    print("Training loss:", training_score)


# Load train and test data
train = pd.read_csv("../data/train.csv", index_col=0)
X_test = pd.read_csv("../data/test.csv", index_col=0)
X_train = train.drop("SalePrice", axis=1)
y_train = np.log(train["SalePrice"])

# Get categorical features
categorical = [
    column for column in train.columns if not is_numeric_dtype(train[column])
]
cat_indices = [
    i for i, column in enumerate(train.columns) if not is_numeric_dtype(train[column])
]

# Encode categorical data
ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=np.nan
)

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

for col in categorical:
    X_train[col].fillna("NaN", inplace=True)
    X_test[col].fillna("NaN", inplace=True)

X_train[categorical] = ordinal_encoder.fit_transform(X_train[categorical])
X_test[categorical] = ordinal_encoder.transform(X_test[categorical])

X_train[X_train_copy.isnull()] = np.nan
X_test[X_test_copy.isnull()] = np.nan


# Tune max_iters (number of trees)
params = {"categorical_features": cat_indices, "scoring": "neg_root_mean_squared_error"}
max_iters_param_grid = {"max_iter": range(20, 120, 10)}
hgbr_1 = HistGradientBoostingRegressor(**params)
g_search = GridSearchCV(hgbr_1, max_iters_param_grid)
_ = g_search.fit(X_train, y_train)

params["max_iter"] = g_search.best_params_["max_iter"]

# Tune tree-specific parameters (max_depth, min_samples_leaf)
tree_features_param_grid = {
    "max_depth": range(4, 10),
    "min_samples_leaf": range(10, 80, 5),
}
hgbr_2 = HistGradientBoostingRegressor(**params)
r_search = RandomizedSearchCV(hgbr_2, param_distributions=tree_features_param_grid)
_ = r_search.fit(X_train, y_train)

params["max_depth"], params["min_samples_leaf"] = (
    r_search.best_params_["max_depth"],
    r_search.best_params_["min_samples_leaf"],
)

# Best model with optimized hyperparameters
best_model = HistGradientBoostingRegressor(**params)
_ = best_model.fit(X_train, y_train)


log_test_preds = best_model.predict(X_test)
test_preds = np.exp(log_test_preds)

submission_name = "hgbr"
submission_string = "../submissions/" + submission_name + "_submission.csv"

result_df = pd.DataFrame({"Id": X_test.index, "SalePrice": test_preds})
result_df.to_csv(submission_string, index=False)
