import joblib
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tarfile

from pandas.plotting import scatter_matrix
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()


def get_insights(data=housing):
    print(data.head(), end="\n\n")
    print(data.info(), end="\n\n")
    print(data.describe(), end="\n\n")
    data.hist(bins=50, figsize=(15, 12))
    plt.show()


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash_function):
    return hash_function(np.int64(identifier)).digest()[-1] < 256 * test_ratio


split_object = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set, strat_test_set = pd.DataFrame(), pd.DataFrame()

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

for train_index, test_index in split_object.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


def draw_heatmap(data=housing):
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=data["population"] / 100, label="population",
              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()


rooms_ix = 3
bedrooms_ix = 4
population_ix = 5
household_ix = 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class LogScaler(BaseEstimator, TransformerMixin):
    bases = {"e": np.log, "2": np.log2, "10": np.log10}

    def __init__(self, base="e"):
        if not (base in self.bases):
            raise ValueError("No logarithm with base {}".format(base))
        self.func = self.bases[base]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.apply_along_axis(self.func, 1, X)


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

lognormal_features = ["median_income", "households", "population", "bedrooms_per_room",
                      "rooms_per_household", "total_rooms", "total_bedrooms"]

caucasus_mountains_features = ["longitude", "latitude", "housing_median_age"]

# Actually the important ones are "INLAND"
categorical_features = ["ocean_proximity"]

num_attribs = list(housing.drop("ocean_proximity", axis=1))
housing["INLAND"] = np.where(housing["ocean_proximity"] == "INLAND", 1.0, 0.0)

longitude_biniraze_pipeline = Pipeline([
    ("selector", DataFrameSelector(["longitude"])),
    ("imputer", SimpleImputer(strategy="median")),
    ("binner", preprocessing.KBinsDiscretizer(n_bins=10, encode="onehot-dense"))
])

latitude_biniraze_pipeline = Pipeline([
    ("selector", DataFrameSelector(["latitude"])),
    ("imputer", SimpleImputer(strategy="median")),
    ("binner", preprocessing.KBinsDiscretizer(n_bins=10, encode="onehot-dense"))
])

housing_median_age_biniraze_pipeline = Pipeline([
    ("selector", DataFrameSelector(["housing_median_age"])),
    ("imputer", SimpleImputer(strategy="median")),
    ("binner", preprocessing.KBinsDiscretizer(n_bins=10, encode="onehot-dense"))
])

scale_pipeline = Pipeline([
    ("selector", DataFrameSelector(lognormal_features)),
    ("attrib adder", CombinedAttributesAdder()),
    ("log scaler", LogScaler()),
    ("std scaler", preprocessing.StandardScaler())
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector("INLAND"))
])

full_pipeline = FeatureUnion(transformer_list=[
    ("cat pipeline", cat_pipeline),
    ("scale pipeline", scale_pipeline),
    ("longitude pipeline", longitude_biniraze_pipeline),
    ("latitude pipeline", latitude_biniraze_pipeline),
    ("housing median age pipeline", latitude_biniraze_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)

print(housing.shape)
# 48425.078937235696 - prev score

# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
#
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print("Linear regression rmse:", lin_rmse)
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print("Decision tree regression rmse:", tree_rmse)

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# display_scores(tree_rmse_scores)
# forest_reg = RandomForestRegressor()
# print("learning...")
# forest_reg.fit(housing_prepared, housing_labels)
# print("validating...")
# scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-scores)
#
# display_scores(forest_rmse_scores)
#
# joblib.dump(forest_reg, "forest_regressor.pkl")

# param_grid = [
#     {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
#     {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
# ]
#
# forest_reg = RandomForestRegressor()
#
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
# grid_search.fit(housing_prepared, housing_labels)

# forest_reg = joblib.load("forest_regressor.pkl")
#
# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()
#
# get_insights()
